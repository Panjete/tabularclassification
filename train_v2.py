import numpy
import gensim
import pandas
import datefinder
import json
import torch
import random

from utils import retrieve_text, generateW2Vembeddings, loadW2Vembeddings,\
                     tokenise_sentence, get_embeddings, read_queries,\
                          find_col, generateFTembed, loadFTembed
from model import Model
from torch.nn.functional import cosine_similarity

TOKENFILE = False
W2VEMBEDD = True
EPOCHS    = 100

def train(trainfile, valfile, interim_text_file = "data/interim_text_file"): ## Change this


    ## Tokenise words and save them as sentences in interim_text_file

    if TOKENFILE:
        retrieve_text(trainfile, interim_text_file)
    print("Retrieved text, tokenized and stored in interim text file")

    ## generate embeddings
    print("Generating embeddings")
    if W2VEMBEDD:
        #generateW2Vembeddings(interim_text_file, save_file="data/w2v")
        generateFTembed(interim_text_file, save_file="data/ft")

    if torch.cuda.is_available():
        device = "cuda"
        print("Cuda is available")
    else:
        device = "cpu"
    print("Generated embeddings")
    ## Define Model, Loss and Optimiser
    model = Model(3).to(device)

    loss_function = torch.nn.CosineEmbeddingLoss() 
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)
    print("Defined Model, Loss and Optim")
    running_loss = 0.0
    ## Train Model
    linenum = 0
    labels = [1, -1, -1, -1, -1, -1]
    labels = torch.tensor(labels).to(device)
    best_accuracy = -1.0

    w2v_embeddings = loadFTembed(save_file="data/ft")#loadW2Vembeddings(save_file="data/w2v")
    for epoch in range(EPOCHS):
        linenum = 0
        model.train()
        with open(trainfile, 'r+') as file:
            for line in file:
                data = json.loads(line) # Load the string in this line of the file
                question = data.get("question", "")
                question_tokens = tokenise_sentence(question)          ## List of tokens
                cols = data.get("table", {}).get("cols", [])
                correct_col = data.get("label_col", ["NULL"])[0]      ## Uniquely One column correct
                #correct_col_number
                training_embs = [correct_col]
                while(len(training_embs)< 6):
                    rand_int = random.randint(0, len(cols)-1)
                    if cols[rand_int] != correct_col:
                        training_embs.append(cols[rand_int])

                #print("Question = ", question, " Training embs = ", training_embs)
                training_embs = [tokenise_sentence(col) for col in training_embs] ## List of List of tokens
            
                training_embs_tensors = []
                for entry in training_embs:
                    total_embedding = torch.tensor(get_embeddings(entry[0], w2v_embeddings))
                    for word in entry[1:]:
                        total_embedding += torch.tensor(get_embeddings(word, w2v_embeddings))
                
                    training_embs_tensors.append(total_embedding)
                training_embs_tensor = torch.stack(training_embs_tensors, dim = 0).to(device)  # 6 * 100 tensor
                print("Train Embs Tensor shape = ", training_embs_tensor.shape)

                ## Bound question length by 20
                # if(len(question_tokens) >= 20):
                #      question_tokens = question_tokens[:20]
                # else:
                #      question_tokens += ["PADDING" for _ in range(20-len(question_tokens))]
                #print("Number of tokens in question = ", len(question_tokens))
                question_tensors = [torch.tensor(get_embeddings(word, w2v_embeddings)) for word in question_tokens]
                question_tensor = torch.stack(question_tensors, dim = 0).to(device)
                print("question Tensor shape = ", question_tensor.shape)
                ### Bring CLS token closer to correct_col embedding
                ### And farther from other negative samples
                CLS = model(question_tensor)
                #print("Model Output shape = ", CLS.shape)
                CLS_broadcasted = CLS.repeat(6, 1)
                #print("Model Output Broadcased shape = ", CLS_broadcasted.shape)

                loss = loss_function(CLS_broadcasted, training_embs_tensor, labels)
                loss.backward()
                optim.step()
                running_loss += loss.item()
                linenum += 1
                if linenum%5000 == 0:
                    print("loss in iter {}, epoch {} = {}".format(linenum, epoch, running_loss/5000))
                    running_loss = 0.0


        #### VALIDATION PHASE #############
        model.eval()
        queries, numqueries = read_queries(valfile) ## Reads just columns

        ## Load Models
        #answer_columns = [] # List of numbers
        #answer_rows    = [] # List of List of numbers
        correct_answers = 0

        for query_num in range(numqueries):
            ## Query Model1 for column
            question_tokens = queries['questions'][query_num]
            question_tensors = [torch.tensor(get_embeddings(word, w2v_embeddings)) for word in question_tokens]
            question_tensor = torch.stack(question_tensors, dim = 0).to(device) ## (Len_question * 100) tensor
            correct_col_number = queries['correct_col_numbers'][query_num]      ## Uniquely One column correct


            columns = queries['table_cols'][query_num]      ## List of List of tokens
            num_columns_this_query = len(columns)
            col_embs_tensors = []
            for entry in columns:                           ## List of tokens
                total_embedding = torch.tensor(get_embeddings(entry[0], w2v_embeddings))
                for word in entry[1:]:
                    total_embedding += torch.tensor(get_embeddings(word, w2v_embeddings))
            
                col_embs_tensors.append(total_embedding)       ## Embedding for this column
            col_embs = torch.stack(col_embs_tensors, dim = 0).to(device)  ## (#columns * 100) tensor embedding, for all columns

            CLS = model(question_tensor)                           ## (1 * 100) tensor
            CLS_broadcast = CLS.repeat(num_columns_this_query, 1)      ## (Len_question * 100) tensor
            similarities = cosine_similarity(CLS_broadcast, col_embs)  ## (Len_question, 1) tensor
            
            max_similarity_column = torch.argmax(similarities, dim=0).item()
            if max_similarity_column == correct_col_number:
                correct_answers += 1
            #answer_columns.append(max_similarity_column)
            #answer_rows.append([0])
            ## Read in appropriate columns
        current_accuracy = correct_answers/numqueries
        print("Val Accuracy at the end of epoch {} is {}".format(epoch, current_accuracy))
        if best_accuracy > current_accuracy:
            torch.save(model.state_dict(), 'data/model.pth')#'/kaggle/working/model.pth')
            best_accuracy = current_accuracy
                            
        
                
    
    return

train("data/A2_train.jsonl", "data/A2_val.jsonl")