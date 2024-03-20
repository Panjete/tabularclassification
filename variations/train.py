import gensim
import json
import torch


### Passing (question, sep1, c1, sep, c2, sep, c3, sep, ...)
### Have a basic MLP at the end, return


from utils import retrieve_text, read_queries,\
                          find_col, pad_tokens, get_embeddings, tokenise_punc_sentence#, generateFTembed, loadFTembed, load_glove , generateW2Vembeddings, loadW2Vembeddings

from redundant.model_bilstm_all import Model
from models_row_sel    import ModelRowSel

import gensim.downloader 
#embeddings = gensim.downloader.load('fasttext-wiki-news-subwords-300')

from combModels import CM

TRAIN_COL = True
TRAIN_ROW = False


EPOCHS    = 100
QUESTION_ANSWER_DELIMITER = ["|"] ## Demarcates question and answer
ENTRY_ROW_WISE_STOP       = ["%"] ## Demarcates multiple elements of a row
TOKEN_LIMIT = 48
ROW_TOKENS  = 64

def train(trainfile, valfile):
    embeddings = gensim.downloader.load('glove-wiki-gigaword-100')
    if torch.cuda.is_available():
        device = "cuda"
        print("Cuda is available")
    else:
        device = "cpu"

    print("Starting Training")

    if TRAIN_COL:

        model = Model(3).to(device)
        print("Starting Col Training, #params = ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
        linenum       = 0
        optim         = torch.optim.Adam(model.parameters(), lr=4e-5)
        best_accuracy = -1.0
        running_loss   = 0.0
        loss_function = torch.nn.HuberLoss().to(device) #torch.nn.BCELoss().to(device)
        print("Defined Model, Loss and Optim")
        
        
        queries, numqueries = read_queries(valfile) ## Reads just columns
        ####### ALL INITIALISATIONS DONE ######## 

        
        for epoch in range(EPOCHS):
            linenum = 0
            model.train()
            with open(trainfile, 'r') as file:
                for line in file:
                    if linenum%16 == 0:
                        # BatchStart == True
                        if linenum != 0:
                            ## Compute total loss per Batch 
                            #questions = numpy.array([numpy.array([get_embeddings(word, embeddings) for word in question]) for question in questions])
                            questions_f = []
                            for question in questions:
                                question   = pad_tokens(question, max_question_token_length)
                                question_f = [torch.tensor(get_embeddings(token, embeddings), requires_grad=False, dtype=torch.float32, device=device) for token in question]
                                question_f = torch.stack(question_f, dim=0)
                                questions_f.append(question_f)

                            final_tensor = torch.stack(questions_f, dim=0)
                            true_label_t = torch.tensor(true_label, device = device, requires_grad=False, dtype=torch.float32).unsqueeze(dim=1)

                            #print("FINAL, LABEL CONSTRUCTED TENSOR SIZE = ", final_tensor.size(), true_label_t.size())
                            pred = model(final_tensor)
                            #print("Model    Predicts : ", pred.size())
                            #print("PRED = ", pred, "true label = ", true_label)
                            loss       = loss_function(pred, true_label_t)
                            loss.backward()
                            optim.step()
                            running_loss += loss.item()
                            #print("loss = ", loss.item(), 'computed when linenum = {}'.format(linenum))
                            questions  = []
                            true_label = []
                            max_question_token_length = 0
                            
                        else:
                            questions  = []
                            true_label = []
                            max_question_token_length = 0

                    data = json.loads(line) # Load the string in this line of the file
                    question = data.get("question", "")
                    question_tokens = tokenise_punc_sentence(question) + QUESTION_ANSWER_DELIMITER          ## List of tokens
                    cols = data.get("table", {}).get("cols", [])
                    correct_col = data.get("label_col", ["NULL"])[0]      ## Uniquely One column correct
                
                    for i, col in enumerate(cols):
                        if col == correct_col:
                            true_label.append(i + 0.0)
        
                        question_tokens += ENTRY_ROW_WISE_STOP + tokenise_punc_sentence(col)  ##List of tokens
                    #final_question_tokens = pad_tokens(question_tokens, TOKEN_LIMIT)
                    max_question_token_length = max(max_question_token_length, len(question_tokens))
                    questions.append(question_tokens)
                        
                        #print("INCORRECT COL TOKEN = ", training_embs)
                        #print("QUESTION TOKENS = ", question_tokens)
                    linenum += 1
                    if linenum%5000 == 0:
                        print("loss in iter {}, epoch {} = {}".format(linenum, epoch, 16*running_loss/5000))
                        running_loss = 0.0


            #### VALIDATION PHASE #############
            
            ## Load Models
            correct_answers = 0
            count_pred = {}
            count_gold = {}
            with torch.no_grad():
                model.eval()
                for query_num in range(numqueries):
                    ## Query Model1 for column
                    question_tokens    = queries['questions'][query_num] + QUESTION_ANSWER_DELIMITER
                    correct_col_number = queries['correct_col_numbers'][query_num]      ## Uniquely One column correct, 0 indexed

                    columns      = queries['table_cols'][query_num]      ## List of List of tokens
                    numcols      = len(columns)
                    for col in columns:
                        question_tokens  += ENTRY_ROW_WISE_STOP + col
                
                    token_tensors    = [torch.tensor(get_embeddings(word, embeddings), device=device, requires_grad=False, dtype=torch.float32) for word in question_tokens]
                    tokens_tensor    = torch.stack(token_tensors, dim = 0)
                    final_tensor     = torch.unsqueeze(tokens_tensor, dim = 0)

                    likely_col = model(final_tensor)

                    max_similarity_column = round(torch.clip(likely_col, 0, numcols-1).item())
                    
            
                    if max_similarity_column == correct_col_number:
                        correct_answers += 1

                    if max_similarity_column in count_pred:
                        count_pred[max_similarity_column] += 1
                    else:
                        count_pred[max_similarity_column] = 1

                    if correct_col_number in count_gold:
                        count_gold[correct_col_number] += 1
                    else:
                        count_gold[correct_col_number] = 1
    
                current_accuracy = correct_answers/numqueries
                
                print("Val Accuracy at the end of epoch {} is {}".format(epoch, current_accuracy))
                print("Distribution Gold = ", count_gold)
                print("Distribution Pred = ", count_pred)
                if best_accuracy < current_accuracy:
                    torch.save(model.state_dict(), 'model_col.pth')#'/kaggle/working/model.pth')
                    best_accuracy = current_accuracy
                    print("Best_accuracy now set to = ", best_accuracy, " and model saved!")

        del model, optim, loss, queries
    ################ COL MODEL TRAINED, MOVE TO ROW MODEL #####################
    if TRAIN_ROW:
        print("Training Row Selector ")

        model_row = ModelRowSel(2).to(device)
        loss_function = torch.nn.HuberLoss().to(device) #torch.nn.BCELoss().to(device)
        optim = torch.optim.Adam(model_row.parameters(), lr=2e-5)
        running_loss = 0.0
        best_accuracy = -1.0

        for epoch in range(EPOCHS):
            linenum = 0
            model_row.train()
            with open(trainfile, 'r') as file:
                for line in file:

                    data = json.loads(line) # Load the string in this line of the file
                    question = data.get("question", "")
                    question_tokens = tokenise_punc_sentence(question) + QUESTION_ANSWER_DELIMITER          ## List of tokens
                    cols = data.get("table", {}).get("cols", [])
                    correct_col = data.get("label_col", ["NULL"])[0]      ## Uniquely One column correct
                    rows  = data.get("table", {}).get("rows", [])
                    num_rows = len(rows)
                    correct_col_number = find_col(correct_col, cols)
                    relevant_col_entries = [row[correct_col_number] for row in rows]
                    correct_rows         = data.get("label_row", ["NULL"])

                    num_batches_rows = (num_rows//32) + 1
                    per_line_tensors = []
                    per_line_label   = []
                    for row_batch in range(num_batches_rows):
                        start_index = 32 * row_batch
                        end_index = min(start_index + 32, num_rows) ## Both not to be included
                        rows_in_this_batch = question_tokens
                        for i in range(start_index, end_index):
                            rows_in_this_batch += ENTRY_ROW_WISE_STOP + tokenise_punc_sentence(relevant_col_entries[i])
                        padded_row = pad_tokens(rows_in_this_batch, ROW_TOKENS)
                        tokenised_rows_batch_tensors= [torch.tensor(get_embeddings(token, embeddings), requires_grad=False, dtype=torch.float32, device=device) for token in padded_row]
                        batch_prepared = torch.stack(tokenised_rows_batch_tensors, dim = 0)
                        per_line_tensors.append(batch_prepared)
                        
                        correct_rows_in_this_batch = []
                        for cor_row in correct_rows:
                            if cor_row >= start_index and cor_row < end_index:
                                correct_rows_in_this_batch.append(cor_row - start_index)
                        if len(correct_rows_in_this_batch) == 0:
                            correct_rows_in_this_batch = [-4.0]
                        label_train = sum(correct_rows_in_this_batch)/len(correct_rows_in_this_batch)
                        per_line_label.append(label_train)

                    true_label_t  = torch.tensor(per_line_label, requires_grad=False, dtype=torch.float32, device=device) 
                    final_tensor  = torch.stack(per_line_tensors, dim=0)
                    #print("Model Input Dim, Label Dim = ", final_tensor.size(), true_label_t.size())

                    predicted_tensor = model_row(final_tensor)
                    #print("Predicted tensor shape = ", final_tensor.size(), true_label_t.size())
                    if predicted_tensor.dim() == 2:
                        true_label_t = true_label_t.unsqueeze(dim=1)

                    loss             = loss_function(predicted_tensor, true_label_t)
                    loss.backward()
                    optim.step()
                    running_loss += loss.item()

                        

                    linenum += 1
                    if linenum%5000 == 0:
                        print("loss in iter {}, epoch {} = {}".format(linenum, epoch, running_loss/5000))
                        running_loss = 0.0


            #### VALIDATION PHASE for ROW MODEL #############
            
            ## Load Models
            correct_answers = 0
            linenum = 0
        
            with torch.no_grad():
                model_row.eval()
                with open(valfile, 'r') as file:
                    for line in file:
                        data = json.loads(line) # Load the string in this line of the file
                        question = data.get("question", "")
                        question_tokens = tokenise_punc_sentence(question) + QUESTION_ANSWER_DELIMITER          ## List of tokens
                        cols = data.get("table", {}).get("cols", [])
                        correct_col = data.get("label_col", ["NULL"])[0]      ## Uniquely One column correct
                        rows  = data.get("table", {}).get("rows", [])
                        num_rows = len(rows)
                        correct_col_number = find_col(correct_col, cols)
                        relevant_col_entries = [row[correct_col_number] for row in rows]
                        correct_rows         = data.get("label_row", ["NULL"])
                        linenum += 1
                        predictions = []
                        num_batches_rows = (num_rows//32) + 1

                        for row_batch in range(num_batches_rows):
                            start_index = 32 * row_batch
                            end_index = min(start_index + 32, num_rows) ## Both not to be included
                            num_rows_in_thi_batch = end_index - start_index
                            rows_in_this_batch = question_tokens
                            for i in range(start_index, end_index):
                                rows_in_this_batch += ENTRY_ROW_WISE_STOP + tokenise_punc_sentence(relevant_col_entries[i])
                           
                            tokenised_rows_batch_tensors= [torch.tensor(get_embeddings(token, embeddings), requires_grad=False, dtype=torch.float32, device=device) for token in rows_in_this_batch]
                            batch_prepared = torch.stack(tokenised_rows_batch_tensors, dim = 0)

                            predicted_tensor = model_row(batch_prepared)
                            prediction = round(torch.clip(predicted_tensor, -20.0, 31.0).item())
                            if prediction > -1 and prediction < num_rows_in_thi_batch:
                                predictions.append(prediction + start_index)



                        if len(predictions) == 0:
                           predictions = [0]
                        if len(predictions) == len(correct_rows):
                            flag = True
                            for p, t in zip(predictions, correct_rows):
                                if p!=t:
                                    flag = False
                                    break
                            if flag:
                                correct_answers += 1
                    
                current_accuracy = correct_answers/linenum
                print("Val Accuracy at the end of epoch {} is {}".format(epoch, current_accuracy))

                if best_accuracy < current_accuracy:
                    torch.save(model_row.state_dict(), 'model_row.pth')
                    best_accuracy = current_accuracy
                    print("Best_accuracy now set to = ", best_accuracy, " and model saved!")

    print("All Trainings Done!")
    cm = CM('model_row.pth', 'model_col.pth')
    cm.save('final.pth')
    return

#train("data/A2_train.jsonl", "data/A2_val.jsonl")#