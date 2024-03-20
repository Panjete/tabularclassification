import numpy
import gensim
import pandas
import datefinder
import json
import torch
from random import randint

### Passing (question, c1)
### Passing (question, c2) ... and so on as the batch
### Have a basic MLP at the end, return


from utils import retrieve_text, generateW2Vembeddings, loadW2Vembeddings,\
                      get_embeddings, read_queries,\
                          find_col, generateFTembed, loadFTembed #, tokenise_sentence
from utils import tokenise_punc_sentence 
from redundant.model_v3 import Model
from torch.nn.functional import cosine_similarity

TOKENFILE = False
W2VEMBEDD = False
EPOCHS    = 100
NS        = 3
QUESTION_ANSWER_DELIMITER = ["|"]

def train(trainfile, valfile, interim_text_file = "data/interim_text_file2"): ## Change this


    ## Tokenise words and save them as sentences in interim_text_file

    if TOKENFILE:
        retrieve_text(trainfile, interim_text_file)
    print("Retrieved text, tokenized and stored in interim text file")

    ## generate embeddings
    print("Generating embeddings")
    if W2VEMBEDD:
        #generateW2Vembeddings(interim_text_file, save_file="data/w2v")
        generateFTembed(interim_text_file, save_file="data/ft3")
    if torch.cuda.is_available():
        device = "cuda"
        print("Cuda is available")
    else:
        device = "cpu"
    print("Generated embeddings")
    ## Define Model, Loss and Optimiser

    model = Model(2).to(device)
    print("NUM PARAMETERS = ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    #loss_function = torch.nn.BCEWithLogitsLoss() #torch.nn.CosineEmbeddingLoss() 
    loss_function = torch.nn.BCELoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Defined Model, Loss and Optim")
    running_loss = 0.0
    ## Train Model
    linenum = 0
    best_accuracy = -1.0

    queries, numqueries = read_queries(valfile) ## Reads just columns

    ####### ALL INITIALISATIONS DONE ######## 

    embeddings = loadFTembed(save_file="data/ft3")#loadW2Vembeddings(save_file="data/w2v")
    #print("sample encodings", list(map(lambda x : get_embeddings(x, w2v_embeddings) , ["row", "5", "73", "age"])))
    #print("positional encodings", model.PE_module.pe[:, :4, :])
    
    for epoch in range(EPOCHS):
        linenum = 0
        model.train()
        with open(trainfile, 'r') as file:
            for line in file:

                if linenum%32 == 0:
                    # BatchStart == True
                    if linenum != 0:
                        ## Compute total loss per Batch 
                        pred       = torch.stack(pred, dim = 0).squeeze()
                        true_label = torch.tensor(true_label, device = device, requires_grad=False, dtype=torch.float32)
                        #print("PRED = ", pred, "true label = ", true_label)
                        loss       = loss_function(pred, true_label)
                        loss.backward()
                        optim.step()
                        running_loss += loss.item()
                        #print("loss = ", loss.item(), 'computed when linenum = {}'.format(linenum))
                        pred = []
                        true_label = []
                        #breakpoint()
                    else:
                        pred = []
                        true_label = []
                # if linenum == 64:
                #     break

                data = json.loads(line) # Load the string in this line of the file
                question = data.get("question", "")
                question_tokens = tokenise_punc_sentence(question)          ## List of tokens
                cols = data.get("table", {}).get("cols", [])
                numcols = len(cols)

                correct_col = data.get("label_col", ["NULL"])[0]      ## Uniquely One column correct
                question_tensors = [torch.tensor(get_embeddings(word, embeddings)) for word in question_tokens]
                
                ### Per Question Positive training
                training_embs         = tokenise_punc_sentence(correct_col) + QUESTION_ANSWER_DELIMITER  ##List of tokens
                training_embs_tensors = [torch.tensor(get_embeddings(word, embeddings)) for word in training_embs]
                final_tensor          = torch.stack(question_tensors + training_embs_tensors, dim = 0).to(device)
                final_tensor.unsqueeze_(dim=0) ## _ indicates in place variant of unsqueeze
                #print("Final Tensor shape : ", final_tensor.size())
                num = model(final_tensor)
                #print("Model (Correct)   Predicts : ", num.size(), num, " correct label = ", correct_label)
                
                pred.append(num)
                true_label.append(1.0)

                #print("CORRECT COL TOKEN = ", training_embs)
                #print("QUESTION TOKENS = ", question_tokens)
                #breakpoint()
                ### Per Col NS training
                for ns in range(randint(1, NS)):
                    col = cols[randint(0, numcols-1)]
                    if col == correct_col:
                        continue
                    training_embs         = tokenise_punc_sentence(col)+ QUESTION_ANSWER_DELIMITER  ##List of tokens
                    training_embs_tensors = [torch.tensor(get_embeddings(word, embeddings)) for word in training_embs]
                    '''
                    ## Bound question length by 20
                    # if(len(question_tokens) >= 20):
                    #      question_tokens = question_tokens[:20]
                    # else:
                    #      question_tokens += ["PADDING" for _ in range(20-len(question_tokens))]
                    #print("Number of tokens in question = ", len(question_tokens))
                    '''
                    final_tensor = torch.stack(question_tensors + training_embs_tensors, dim = 0).to(device)
                    final_tensor.unsqueeze_(dim=0) ## _ indicates in place variant of unsqueeze
            
                    num = model(final_tensor)
                    #print("model_output_size and type = ", num.size(), num.dtype)
                    pred.append(num)
                    true_label.append(0.0)
                    #print("INCORRECT COL TOKEN = ", training_embs)
                    #print("QUESTION TOKENS = ", question_tokens)

                linenum += 1
                if linenum%5000 == 0:
                    print("loss in iter {}, epoch {} = {}".format(linenum, epoch, 32* running_loss/5000))
                    running_loss = 0.0


        #### VALIDATION PHASE #############
        
        ## Load Models
        #answer_columns = [] # List of numbers
        #answer_rows    = [] # List of List of numbers
        correct_answers = 0
        with torch.no_grad():
            model.eval()
            for query_num in range(numqueries):
                # if query_num == 30:
                #     break
                ## Query Model1 for column
                question_tokens    = queries['questions'][query_num]
                correct_col_number = queries['correct_col_numbers'][query_num]      ## Uniquely One column correct, 0 indexed

                similarities = []
                columns      = queries['table_cols'][query_num]      ## List of List of tokens
                for col in columns:
                    question_tensors = [torch.tensor(get_embeddings(word, embeddings)) for word in question_tokens + QUESTION_ANSWER_DELIMITER]
                    col_tensors      = [torch.tensor(get_embeddings(word, embeddings)) for word in col]
                    final_tensor     = torch.stack(question_tensors + col_tensors, dim = 0).to(device) ## (Len_question+col, 100) tensor
                    final_tensor.unsqueeze_(dim=0)
                    likelihood_of_this_col = model(final_tensor).item()
                    similarities.append(likelihood_of_this_col)

                max_similarity_column = numpy.argmax(similarities)
                if max_similarity_column == correct_col_number:
                    correct_answers += 1

                #answer_columns.append(max_similarity_column)
                #answer_rows.append([0])
                ## Read in appropriate columns
            current_accuracy = correct_answers/numqueries
            print("Val Accuracy at the end of epoch {} is {}".format(epoch, current_accuracy))
            if best_accuracy < current_accuracy:
                torch.save(model.state_dict(), 'data/model.pth')#'/kaggle/working/model.pth')
                best_accuracy = current_accuracy
                print("Best_accuracy now set to = ", best_accuracy, " and model saved!")
                
    return

train("data/A2_train.jsonl", "data/A2_val.jsonl")