import gensim
import json
import torch
from random import randint

### Passing (question, c1)
### Passing (question, c2) ... and so on as the batch
### Have a basic MLP at the end, return

import gensim.downloader
from combModels import CM
from torch.nn.functional import cosine_similarity
from utils import get_embeddings, read_queries, pad_tokens, find_col
from models        import Model, ModelRowSel, ModelCompRows, ModelTransformer
from utils import tokenise_punc_sentence, tokenise_punc_sentence_with_number, tokenise_punc_sentence_question

from configs import *


def train_col_model(trainfile, valfile):
    #### <<question> <col>> -> GRU -> Score based model. Incorporates negative sampling.
    embeddings = gensim.downloader.load(EMBEDDINGS)
    ## Tokenise words and save them as sentences in interim_text_file
    if torch.cuda.is_available():
        device = "cuda"
        print("Cuda is available")
    else:
        device = "cpu"

    model_col = Model(MODEL_COL_STACKS).to(device)
    print("Starting Col Training, #params = ", sum(p.numel() for p in model_col.parameters() if p.requires_grad))
    loss_function = torch.nn.BCEWithLogitsLoss().to(device)
    optim         = torch.optim.Adam(model_col.parameters(), lr=LR_COL_MODEL)
    linenum       = 0
    running_loss  = 0.0
    best_accuracy = -1.0
    queries, numqueries = read_queries(valfile) ## Reads just columns

    ####### ALL INITIALISATIONS DONE ######## 
    for epoch in range(EPOCHS):
        linenum = 0
        model_col.train()
        with open(trainfile, 'r') as file:
            for line in file:
                if linenum%16 == 0:
                    # BatchStart == True
                    if linenum != 0:
                        #print("QUESTIONS = ", questions, true_label)
                        #print("MAX LEN = ", max_question_token_length)
                        ## Compute total loss per Batch 
                        #questions = numpy.array([numpy.array([get_embeddings(word, embeddings) for word in question]) for question in questions])
                        questions_f = []
                        for question in questions:
                            question   = pad_tokens(question, max_question_token_length)
                            question_f = [torch.tensor(get_embeddings(token, embeddings), requires_grad=False, dtype=torch.float32, device=device) for token in question]
                            question_f = torch.stack(question_f, dim=0)
                            questions_f.append(question_f)
                        #breakpoint()
                        final_tensor = torch.stack(questions_f, dim=0)
                        true_label_t = torch.tensor(true_label, device = device, requires_grad=False, dtype=torch.float32).unsqueeze(dim=1)

                        #print("FINAL, LABEL CONSTRUCTED TENSOR SIZE = ", final_tensor.size(), true_label_t.size())
                        pred = model_col(final_tensor)
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
                question_tokens = tokenise_punc_sentence(question)          ## List of tokens
                cols = data.get("table", {}).get("cols", [])
                numcols = len(cols)

                correct_col = data.get("label_col", ["NULL"])[0]      ## Uniquely One column correct
                #question_tensors = [torch.tensor(get_embeddings(word, embeddings)) for word in question_tokens]
                ### Per Question Positive training
                training_embs         = QUESTION_ANSWER_DELIMITER + tokenise_punc_sentence(correct_col)  ##List of tokens
                final_positive_tokens = question_tokens + training_embs
                max_question_token_length = max(max_question_token_length, len(final_positive_tokens))
                #final_positive_tokens = pad_tokens(final_positive_tokens, TOKEN_LIMIT)
                
                questions.append(final_positive_tokens)
                true_label.append(1.0)

                #print("CORRECT COL TOKEN = ", training_embs)
                #print("QUESTION TOKENS = ", question_tokens)
                #breakpoint()
                ### Per Col NS training
                for _ in range(NS):
                    col = cols[randint(0, numcols-1)]
                    if col == correct_col:
                        true_label.append(1.0)
                    else:
                        true_label.append(0.0)
                    
                    final_negative_tokens = question_tokens + QUESTION_ANSWER_DELIMITER + tokenise_punc_sentence(col)   ##List of tokens

                    max_question_token_length = max(max_question_token_length, len(final_negative_tokens))
                    #final_negative_tokens = pad_tokens(final_negative_tokens, TOKEN_LIMIT)
                
                    #print("model_output_size and type = ", num.size(), num.dtype)
                    questions.append(final_negative_tokens)
                    
                    #print("INCORRECT COL TOKEN = ", training_embs)
                    #print("QUESTION TOKENS = ", question_tokens)

                linenum += 1
                if linenum%5000 == 0:
                    print("loss in iter {}, epoch {} = {}".format(linenum, epoch, 16*running_loss/5000))
                    running_loss = 0.0


        #### VALIDATION PHASE #############
        
        ## Load Models
        correct_answers = 0
        with torch.no_grad():
            model_col.eval()
            for query_num in range(numqueries):
                ## Query Model1 for column
                question_tokens    = queries['questions'][query_num]
                correct_col_number = queries['correct_col_numbers'][query_num]      ## Uniquely One column correct, 0 indexed

                columns      = queries['table_cols'][query_num]      ## List of List of tokens
                final_tensor = []
                max_init = 0
                for col in columns:
                    max_init = max(max_init, len(col) + 1 + len(question_tokens))
                #print("Max Init for this question = ", max_init)
                # max_question_token_length = []
                for col in columns:
                    this_column_tokens  = pad_tokens(question_tokens + QUESTION_ANSWER_DELIMITER + col, max_init)
                    #print("TOKENS FOR THIS QUESTION VAL = ", this_column_tokens)
                    token_tensors    = [torch.tensor(get_embeddings(word, embeddings), device=device, requires_grad=False, dtype=torch.float32) for word in this_column_tokens]
                    col_tensor       = torch.stack(token_tensors, dim = 0)                  ## (Len_question+col, 100) tensor
                    final_tensor.append(col_tensor)

                final_tensor = torch.stack(final_tensor, dim = 0)
                likelihood_of_cols = model_col(final_tensor)
                #print("Predicted tensors and argmax = ", likelihood_of_cols, torch.argmax(likelihood_of_cols))
                max_similarity_column = torch.argmax(likelihood_of_cols)
                if max_similarity_column == correct_col_number:
                    correct_answers += 1

                #answer_columns.append(max_similarity_column)
                #answer_rows.append([0])
                ## Read in appropriate columns
            current_accuracy = correct_answers/numqueries
            print("Val Accuracy at the end of epoch {} is {}".format(epoch, current_accuracy))
            if best_accuracy < current_accuracy:
                torch.save(model_col.state_dict(), 'model_col.pth')
                best_accuracy = current_accuracy
                print("Best_accuracy now set to = ", best_accuracy, " and model saved!")
                
    return


def train_row_model(trainfile, valfile):
    #### <<question> <trimmed_annotated_row>> -> GRU -> Score based model. Incorporates negative sampling.
    embeddings = gensim.downloader.load(EMBEDDINGS)
    ## Tokenise words and save them as sentences in interim_text_file
    if torch.cuda.is_available():
        device = "cuda"
        print("Cuda is available")
    else:
        device = "cpu"

    model_row = ModelRowSel(MODEL_ROW_STACKS).to(device)
    print("Starting Row Training, #params = ", sum(p.numel() for p in model_row.parameters() if p.requires_grad))
    
    optim         = torch.optim.Adam(model_row.parameters(), lr=LR_COL_MODEL)
    linenum       = 0
    running_loss  = 0.0
    best_accuracy = -1.0
    loss_function = torch.nn.BCEWithLogitsLoss() #torch.nn.CosineEmbeddingLoss().to(device)

    ####### ALL INITIALISATIONS DONE ######## 
    for epoch in range(EPOCHS_ROW):
        linenum = 0
        model_row.train()
        with open(trainfile, 'r') as file:
            for line in file:
                if linenum%16 == 0:
                    # BatchStart == True
                    if linenum != 0:
                        #print("QUESTIONS = ", questions, true_label)
                        #print("MAX LEN = ", max_question_token_length)
                        ## Compute total loss per Batch 
                        #questions = numpy.array([numpy.array([get_embeddings(word, embeddings) for word in question]) for question in questions])
                        # print("PRINTING QUESTIONS AND ANSWERS")
                        # [print("QUESTION", question, "LABLE", label) for question, label in zip(questions, true_label)]
                        # breakpoint()
                        questions_f = []
                        for question in questions:
                            question   = pad_tokens(question, max_question_token_length)
                            question_f = [torch.tensor(get_embeddings(token, embeddings), requires_grad=False, dtype=torch.float32, device=device) for token in question]
                            question_f = torch.stack(question_f, dim=0)
                            questions_f.append(question_f)
                        #breakpoint()
                        final_tensor = torch.stack(questions_f, dim=0)
                        true_label_t = torch.tensor(true_label, device = device, requires_grad=False, dtype=torch.float32).unsqueeze(dim=1)

                        #print("FINAL, LABEL CONSTRUCTED TENSOR SIZE = ", final_tensor.size(), true_label_t.size())
                        pred = model_row(final_tensor)
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
                question_tokens = tokenise_punc_sentence_question(question)          ## List of tokens
                cols = data.get("table", {}).get("cols", [])
                cols_cleaned = [tokenise_punc_sentence_question(col) for col in cols]
                cols_cleaned = [col[:min(2, len(col))] for col in cols_cleaned]
                #numcols = len(cols)
                
                rows = data.get("table", {}).get("rows", [])
                numrows = len(rows)

                correct_col = data.get("label_col", ["NULL"])[0]             ## Uniquely One column correct
                correct_row_number = data.get("label_row", ["NULL"])[0]      ## Assume for now that only one row correct. Anyways true for 90% cases
                correct_row = rows[correct_row_number]                       ## List of tokens
                correct_col_number = find_col(correct_col, cols)
                correct_entry_cell = correct_row[correct_col_number]
                
                ### Per Question Positive training
                final_positive_tokens     = question_tokens + QUESTION_ANSWER_DELIMITER + tokenise_punc_sentence_with_number(correct_row, True, cols_cleaned) #+ ROW_ANSWER_DELIMITER + tokenise_punc_sentence_with_number([correct_entry_cell], False) ##List of tokens
    
                max_question_token_length = max(max_question_token_length, len(final_positive_tokens))
                #final_positive_tokens = pad_tokens(final_positive_tokens, TOKEN_LIMIT)
                
                questions.append(final_positive_tokens)
                true_label.append(1.0)

                #print("CORRECT COL TOKEN = ", training_embs)
                #print("QUESTION TOKENS = ", question_tokens)
                #breakpoint()
                ### Per Col NS training
                for _ in range(NS):
                    row_index = randint(0, numrows-1)
                    if row_index == correct_row_number:
                        true_label.append(1.0)
                    else:
                        true_label.append(0.0)

                    negative_row  = rows[row_index]
                    negative_cell = negative_row[correct_col_number]
                    
                    final_negative_tokens = question_tokens + QUESTION_ANSWER_DELIMITER + tokenise_punc_sentence_with_number(negative_row, True, cols_cleaned) #+ ROW_ANSWER_DELIMITER + tokenise_punc_sentence_with_number([negative_cell], False) ##List of tokens

                    max_question_token_length = max(max_question_token_length, len(final_negative_tokens))
                    #final_negative_tokens = pad_tokens(final_negative_tokens, TOKEN_LIMIT)
                
                    #print("model_output_size and type = ", num.size(), num.dtype)
                    questions.append(final_negative_tokens)
                    
                    #print("INCORRECT COL TOKEN = ", training_embs)
                    #print("QUESTION TOKENS = ", question_tokens)

                linenum += 1
                if linenum%5000 == 0:
                    print("loss in iter {}, epoch {} = {}".format(linenum, epoch, 16*running_loss/5000))
                    running_loss = 0.0


        #### VALIDATION PHASE #############
        
        ## Load Model
        correct_answers = 0
        linenum = 0
        with torch.no_grad():
            model_row.eval()
            with open(valfile, 'r') as file:
                for line in file:
                    if randint(0, 50) != 0: ## Expensive operation, perform selectively
                        continue
        
                    data = json.loads(line)                                    # Load the string in this line of the file
                    question = data.get("question", "")
                    question_tokens = tokenise_punc_sentence_question(question)          ## List of tokens
                    cols = data.get("table", {}).get("cols", [])
                    cols_cleaned = [tokenise_punc_sentence_question(col) for col in cols]
                    cols_cleaned = [col[:min(2, len(col))] for col in cols_cleaned]
                    #numcols = len(cols)
                    
                    rows = data.get("table", {}).get("rows", [])
                    numrows = len(rows)
                    linenum += 1

                    correct_col = data.get("label_col", ["NULL"])[0]             ## Uniquely One column correct
                    correct_row_number = data.get("label_row", ["NULL"])[-1]      ## Assume for now that only one row correct. Anyways true for 90% cases
                    correct_row = rows[correct_row_number]                       ## List of tokens
                    correct_col_number = find_col(correct_col, cols)
                    correct_entry_cell = correct_row[correct_col_number]

                    
                    max_row_score = -1.0
                    max_row_at    = -1
                    for row_index, row in enumerate(rows):
                        cell_for_this_row = row[correct_col_number]
                        this_row_tokens   = question_tokens + tokenise_punc_sentence_with_number(row, True, cols_cleaned) #+ ROW_ANSWER_DELIMITER + tokenise_punc_sentence_with_number([cell_for_this_row], False)
                        token_tensors    = [torch.tensor(get_embeddings(word, embeddings), device=device, requires_grad=False, dtype=torch.float32) for word in this_row_tokens]
                        row_tensor       = torch.stack(token_tensors, dim = 0)                  ## (Len_question+col, 100) tensor
                        
                        likelihood_of_row = model_row(row_tensor).item()

                        if likelihood_of_row > max_row_score:
                            max_row_score = likelihood_of_row
                            max_row_at    = row_index
                    
                    if max_row_at == correct_row_number:
                        correct_answers += 1

                #answer_columns.append(max_similarity_column)
                #answer_rows.append([0])
                ## Read in appropriate columns
            current_accuracy = correct_answers/linenum
            print("Val Accuracy at the end of epoch {} is {}".format(epoch, current_accuracy))
            if best_accuracy < current_accuracy:
                torch.save(model_row.state_dict(), 'model_row.pth')
                best_accuracy = current_accuracy
                print("Best_accuracy now set to = ", best_accuracy, " and model saved!")
                
    return


def train_row_model_trans(trainfile, valfile):

    embeddings = gensim.downloader.load(EMBEDDINGS)
    ## Tokenise words and save them as sentences in interim_text_file
    if torch.cuda.is_available():
        device = "cuda"
        print("Cuda is available")
    else:
        device = "cpu"

    model_row = ModelTransformer(MODEL_ROW_STACKS).to(device)
    print("Starting Row Training, #params = ", sum(p.numel() for p in model_row.parameters() if p.requires_grad))
    
    optim         = torch.optim.Adam(model_row.parameters(), lr=LR_COL_MODEL)
    linenum       = 0
    running_loss  = 0.0
    best_accuracy = -1.0
    loss_function = torch.nn.BCEWithLogitsLoss() #torch.nn.CosineEmbeddingLoss().to(device)

    ####### ALL INITIALISATIONS DONE ######## 
    for epoch in range(EPOCHS_ROW):
        linenum = 0
        model_row.train()
        with open(trainfile, 'r') as file:
            for line in file:
                if linenum%16 == 0:
                    # BatchStart == True
                    if linenum != 0:
                        #print("QUESTIONS = ", questions, true_label)
                        #print("MAX LEN = ", max_question_token_length)
                        ## Compute total loss per Batch 
                        #questions = numpy.array([numpy.array([get_embeddings(word, embeddings) for word in question]) for question in questions])
                        # print("PRINTING QUESTIONS AND ANSWERS")
                        # [print("QUESTION", question, "LABLE", label) for question, label in zip(questions, true_label)]
                        # breakpoint()
                        questions_f = []
                        max_question_token_length = min(max_question_token_length, 300)
                        for question in questions:
                            question   = pad_tokens(question, max_question_token_length)
                            question_f = [torch.tensor(get_embeddings(token, embeddings), requires_grad=False, dtype=torch.float32, device=device) for token in question]
                            question_f = torch.stack(question_f, dim=0)
                            questions_f.append(question_f)
                        #breakpoint()
                        final_tensor = torch.stack(questions_f, dim=0)
                        true_label_t = torch.tensor(true_label, device = device, requires_grad=False, dtype=torch.float32).unsqueeze(dim=1)

                        #print("FINAL, LABEL CONSTRUCTED TENSOR SIZE = ", final_tensor.size(), true_label_t.size())
                        pred = model_row(final_tensor)
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
                question_tokens = tokenise_punc_sentence_question(question)          ## List of tokens
                cols = data.get("table", {}).get("cols", [])
                cols_cleaned = [tokenise_punc_sentence_question(col) for col in cols]
                cols_cleaned = [col[:min(2, len(col))] for col in cols_cleaned]
                #numcols = len(cols)
                
                rows = data.get("table", {}).get("rows", [])
                numrows = len(rows)

                correct_col = data.get("label_col", ["NULL"])[0]             ## Uniquely One column correct
                correct_row_number = data.get("label_row", ["NULL"])[0]      ## Assume for now that only one row correct. Anyways true for 90% cases
                correct_row = rows[correct_row_number]                       ## List of tokens
                correct_col_number = find_col(correct_col, cols)
                
                ### Per Question Positive training
                final_positive_tokens     = question_tokens + QUESTION_ANSWER_DELIMITER + tokenise_punc_sentence_with_number(correct_row, True, cols_cleaned) #+ ROW_ANSWER_DELIMITER + tokenise_punc_sentence_with_number([correct_entry_cell], False) ##List of tokens
    
                max_question_token_length = max(max_question_token_length, len(final_positive_tokens))
                #final_positive_tokens = pad_tokens(final_positive_tokens, TOKEN_LIMIT)
                
                questions.append(final_positive_tokens)
                true_label.append(1.0)

                #print("CORRECT COL TOKEN = ", training_embs)
                #print("QUESTION TOKENS = ", question_tokens)
                #breakpoint()
                ### Per Col NS training
                for _ in range(NS):
                    row_index = randint(0, numrows-1)
                    if row_index == correct_row_number:
                        true_label.append(1.0)
                    else:
                        true_label.append(0.0)

                    negative_row  = rows[row_index]
                    negative_cell = negative_row[correct_col_number]
                    
                    final_negative_tokens = question_tokens + QUESTION_ANSWER_DELIMITER + tokenise_punc_sentence_with_number(negative_row, True, cols_cleaned) #+ ROW_ANSWER_DELIMITER + tokenise_punc_sentence_with_number([negative_cell], False) ##List of tokens

                    max_question_token_length = max(max_question_token_length, len(final_negative_tokens))
                    #final_negative_tokens = pad_tokens(final_negative_tokens, TOKEN_LIMIT)
                
                    #print("model_output_size and type = ", num.size(), num.dtype)
                    questions.append(final_negative_tokens)
                    
                    #print("INCORRECT COL TOKEN = ", training_embs)
                    #print("QUESTION TOKENS = ", question_tokens)

                linenum += 1
                if linenum%5000 == 0:
                    print("loss in iter {}, epoch {} = {}".format(linenum, epoch, 16*running_loss/5000))
                    running_loss = 0.0


        #### VALIDATION PHASE #############
        
        ## Load Model
        correct_answers = 0
        linenum = 0
        with torch.no_grad():
            model_row.eval()
            with open(valfile, 'r') as file:
                for line in file:
                    if randint(0, 100) != 0: ## Expensive operation, perform selectively
                        pass
        
                    data = json.loads(line)                                    # Load the string in this line of the file
                    question = data.get("question", "")
                    question_tokens = tokenise_punc_sentence_question(question)          ## List of tokens
                    cols = data.get("table", {}).get("cols", [])
                    cols_cleaned = [tokenise_punc_sentence_question(col) for col in cols]
                    cols_cleaned = [col[:min(2, len(col))] for col in cols_cleaned]
                    #numcols = len(cols)
                    
                    rows = data.get("table", {}).get("rows", [])
                    numrows = len(rows)
                    linenum += 1

                    correct_col = data.get("label_col", ["NULL"])[0]             ## Uniquely One column correct
                    correct_row_number = data.get("label_row", ["NULL"])[-1]      ## Assume for now that only one row correct. Anyways true for 90% cases
                    correct_row = rows[correct_row_number]                       ## List of tokens
                    correct_col_number = find_col(correct_col, cols)
                    
                    max_row_score = -1.0
                    max_row_at    = -1
                    for row_index, row in enumerate(rows):
                        
                        this_row_tokens   = question_tokens + tokenise_punc_sentence_with_number(row, True, cols_cleaned) #+ ROW_ANSWER_DELIMITER + tokenise_punc_sentence_with_number([cell_for_this_row], False)
                        this_row_tokens   = pad_tokens(this_row_tokens, min(len(this_row_tokens), 300))
                        token_tensors    = [torch.tensor(get_embeddings(word, embeddings), device=device, requires_grad=False, dtype=torch.float32) for word in this_row_tokens]
                        row_tensor       = torch.stack(token_tensors, dim = 0)                  ## (Len_question+col, 100) tensor
                        
                        likelihood_of_row = model_row(row_tensor).item()

                        if likelihood_of_row > max_row_score:
                            max_row_score = likelihood_of_row
                            max_row_at    = row_index
                    
                    if max_row_at == correct_row_number:
                        correct_answers += 1

                #answer_columns.append(max_similarity_column)
                #answer_rows.append([0])
                ## Read in appropriate columns
            current_accuracy = correct_answers/linenum
            print("Val Accuracy at the end of epoch {} is {}".format(epoch, current_accuracy))
            if best_accuracy < current_accuracy:
                torch.save(model_row.state_dict(), 'model_row.pth')
                best_accuracy = current_accuracy
                print("Best_accuracy now set to = ", best_accuracy, " and model saved!")
                
    return

def train_row_model_similarity(trainfile, valfile):
    ### <question> -> GRU \
    ###                    -> similarity score based model.
    ### <row>      -> GRU /
    embeddings = gensim.downloader.load(EMBEDDINGS)
    if torch.cuda.is_available():
        device = "cuda"
        print("Cuda is available")
    else:
        device = "cpu"

    model_row = ModelCompRows(MODEL_ROW_STACKS).to(device)
    print("Starting Row Training, #params = ", sum(p.numel() for p in model_row.parameters() if p.requires_grad))
    
    optim         = torch.optim.Adam(model_row.parameters(), lr=LR_COL_MODEL)
    linenum       = 0
    running_loss  = 0.0
    best_accuracy = -1.0
    loss_function = torch.nn.CosineEmbeddingLoss().to(device) #torch.nn.CosineEmbeddingLoss().to(device)

    ####### ALL INITIALISATIONS DONE ######## 
    for epoch in range(EPOCHS_ROW):
        linenum = 0
        model_row.train()
        with open(trainfile, 'r') as file:
            for line in file:
    
                if linenum%16 == 0:
                    # BatchStart == True
                    if linenum != 0:
                        print("PRINTING QUESTIONS AND ANSWERS")
                        [print(question) for question in questions_answers]
                        print("PRINTING ROWS")
                        [print(row) for row in prospective_rows]
                        breakpoint()
                        #print("QUESTIONS = ", questions, true_label)
                        #print("MAX LEN = ", max_question_token_length)
                        ## Compute total loss per Batch 
                        #questions = numpy.array([numpy.array([get_embeddings(word, embeddings) for word in question]) for question in questions])
                        questions_f = []
                        for question in questions_answers:
                            question   = pad_tokens(question, max_question_token_length)
                            question_f = [torch.tensor(get_embeddings(token, embeddings), requires_grad=False, dtype=torch.float32, device=device) for token in question]
                            question_f = torch.stack(question_f, dim=0)
                            questions_f.append(question_f)
                        #breakpoint()
                            
                        rows_f = []
                        for row in prospective_rows:
                            row   = pad_tokens(row, max_row_token_length)
                            row_f = [torch.tensor(get_embeddings(token, embeddings), requires_grad=False, dtype=torch.float32, device=device) for token in row]
                            row_f = torch.stack(row_f, dim=0)
                            rows_f.append(row_f)
                        
                        final_questions_tensor = torch.stack(questions_f, dim=0)
                        final_rows_tensor      = torch.stack(rows_f     , dim=0)
                        true_label_t = torch.tensor(true_label, device = device, requires_grad=False, dtype=torch.float32)


                        #print("FINAL, LABEL CONSTRUCTED TENSOR SIZE = ", final_tensor.size(), true_label_t.size())
                        pred1, pred2 = model_row(final_questions_tensor, final_rows_tensor)

                        #print("Model    Predicts : ", pred1.size(), pred2.size())
                        #print("PRED = ", pred, "true label = ", true_label)
                        loss   = loss_function(pred1[0, :], pred2[0, :], true_label_t[0])
                        for i in range(1, 16 * (1+NS)):
                            loss += loss_function(pred1[i, :], pred2[i, :], true_label_t[i])
                        loss.backward()
                        optim.step()
                        running_loss += loss.item()
                        #print("loss = ", loss.item(), 'computed when linenum = {}'.format(linenum))
                        questions_answers  = []
                        prospective_rows   = []
                        true_label = []
                        max_question_token_length = 0
                        max_row_token_length      = 0
                        
                    else:
                        questions_answers  = []
                        prospective_rows   = []
                        true_label = []
                        max_question_token_length = 0
                        max_row_token_length      = 0

                data = json.loads(line) # Load the string in this line of the file
                question_of_this_line = data.get("question", "")
                question_tokens = tokenise_punc_sentence_question(question_of_this_line)          ## List of tokens
                cols = data.get("table", {}).get("cols", [])
                #numcols = len(cols)
                
                rows = data.get("table", {}).get("rows", [])
                numrows = len(rows)

                correct_col = data.get("label_col", ["NULL"])[0]             ## Uniquely One column correct
                correct_col_number = find_col(correct_col, cols)

                correct_row_number = data.get("label_row", ["NULL"])[-1]      ## Assume for now that only one row correct. Anyways true for 90% cases
                correct_row        = rows[correct_row_number]                       ## List of tokens
                
                correct_entry_cell = correct_row[correct_col_number]
                
                question_with_correc_cell_tokens = question_tokens + QUESTION_ANSWER_DELIMITER + tokenise_punc_sentence_with_number([correct_entry_cell], False)
                max_question_token_length = max(max_question_token_length, len(question_with_correc_cell_tokens))
                questions_answers.append(question_with_correc_cell_tokens)

                row_tokens_for_correct_row = tokenise_punc_sentence_with_number(correct_row, True)
                max_row_token_length      = max(max_row_token_length, len(row_tokens_for_correct_row))
                prospective_rows.append(row_tokens_for_correct_row)
                ### Per Question Positive training
                true_label.append(1.0)

                #breakpoint()
                ### Per Col NS training
                for _ in range(NS):
                    row_index = randint(0, numrows-1)
                    if row_index == correct_row_number:
                        true_label.append(1.0)
                    else:
                        true_label.append(-1.0)

                    negative_row  = rows[row_index]
                    negative_cell = negative_row[correct_col_number]
                    
                    question_with_this_cell_tokens = question_tokens + QUESTION_ANSWER_DELIMITER + tokenise_punc_sentence_with_number([negative_cell], False)
                    max_question_token_length = max(max_question_token_length, len(question_with_this_cell_tokens))
                    questions_answers.append(question_with_this_cell_tokens)

                    row_tokens_for_this_row = tokenise_punc_sentence_with_number(negative_row, True)
                    max_row_token_length      = max(max_row_token_length, len(row_tokens_for_this_row))
                    prospective_rows.append(row_tokens_for_this_row)

                linenum += 1
                if linenum%5000 == 0:
                    print("loss in iter {}, epoch {} = {}".format(linenum, epoch, 16*running_loss/5000))
                    running_loss = 0.0


        #### VALIDATION PHASE #############
        
        ## Load Model
        correct_answers = 0
        linenum = 0
        with torch.no_grad():
            model_row.eval()
            with open(valfile, 'r') as file:
                for line in file:
                    if randint(0, 50) != 0: ## Expensive operation, perform selectively
                        pass
        
                    data = json.loads(line)                                    # Load the string in this line of the file
                    question = data.get("question", "")
                    question_tokens = tokenise_punc_sentence_question(question)          ## List of tokens
                    cols = data.get("table", {}).get("cols", [])
                    #numcols = len(cols)
                    
                    rows = data.get("table", {}).get("rows", [])
                    numrows = len(rows)
                    linenum += 1

                    correct_col = data.get("label_col", ["NULL"])[0]             ## Uniquely One column correct
                    correct_row_number = data.get("label_row", ["NULL"])[-1]      ## Assume for now that only one row correct. Anyways true for 90% cases
                    correct_row = rows[correct_row_number]                       ## List of tokens
                    correct_col_number = find_col(correct_col, cols)
                    correct_entry_cell = correct_row[correct_col_number]
                    
                    max_row_score = -1.0
                    max_row_at    = -1

                    if numrows%32 == 0:
                        num_batches_rows = (numrows//32)
                    else:
                        num_batches_rows = (numrows//32) + 1

                    for row_batch in range(num_batches_rows):
                        start_index = 32 * row_batch
                        end_index = min(start_index + 32, numrows) ## Both not to be included
                        rows_in_this_batch = end_index - start_index
                        questions_answers = []
                        prospective_rows  = []
                        max_row_length_in_this_batch = 0
                        max_question_token_length    = 0
                        for index in range(start_index, end_index):
                            row_tokens_for_this_row = tokenise_punc_sentence_with_number(rows[index], True)
                            max_row_length_in_this_batch = max(max_row_length_in_this_batch, len(row_tokens_for_this_row))
                            prospective_rows.append(row_tokens_for_this_row)

                            this_row_entry  = rows[index][correct_col_number]
                    
                            question_with_this_cell_tokens = question_tokens + QUESTION_ANSWER_DELIMITER + tokenise_punc_sentence_with_number([this_row_entry], False)
                            max_question_token_length = max(max_question_token_length, len(question_with_this_cell_tokens))
                            questions_answers.append(question_with_this_cell_tokens)
                            
                        rows_f = []
                        for row in prospective_rows:
                            row   = pad_tokens(row , max_row_length_in_this_batch)
                            row_f = [torch.tensor(get_embeddings(token, embeddings), requires_grad=False, dtype=torch.float32, device=device) for token in row]
                            row_f = torch.stack(row_f, dim=0)
                            rows_f.append(row_f)

                        questions_f = []
                        for question in questions_answers:
                            question   = pad_tokens(question, max_question_token_length)
                            question_f = [torch.tensor(get_embeddings(token, embeddings), requires_grad=False, dtype=torch.float32, device=device) for token in question]
                            question_f = torch.stack(question_f, dim=0)
                            questions_f.append(question_f)

                        final_rows_tensor      = torch.stack(rows_f     , dim=0)
                        final_questions_tensor = torch.stack(questions_f, dim=0)
                        pred1, pred2 = model_row(final_questions_tensor, final_rows_tensor)
                        #print("pred1 pred2 size == ", pred1.size(), pred2.size())
                        similarities_in_this_batch = cosine_similarity(pred1, pred2)
                        #print("Similarity tensor : ", similarities_in_this_batch)
                        max_score_in_this_batch, max_score_at_in_this_batch = torch.max(similarities_in_this_batch, dim=0)
                        #print("max value : ", max_score_in_this_batch.item())
                        # = torch.argmax(similarities_in_this_batch).item()
                        #print("armgmax : ", max_score_at_in_this_batch.item())

                        if  max_score_in_this_batch.item() > max_row_score:
                            max_row_score = max_score_in_this_batch.item()
                            max_row_at    = max_score_at_in_this_batch.item() + start_index
                    
                        
                    if max_row_at == correct_row_number:
                        correct_answers += 1

                #answer_columns.append(max_similarity_column)
                #answer_rows.append([0])
                ## Read in appropriate columns
            current_accuracy = correct_answers/linenum
            print("Val Accuracy at the end of epoch {} is {}".format(epoch, current_accuracy))
            if best_accuracy < current_accuracy:
                torch.save(model_row.state_dict(), 'model_row.pth')
                best_accuracy = current_accuracy
                print("Best_accuracy now set to = ", best_accuracy, " and model saved!")
                
    return


def train_row_model(trainfile, valfile):
    #### <question> <row1> <row2> <row3> ... -> GRU ->  predict index of the correct column
    print("Training Row Selector ")
    embeddings = gensim.downloader.load(EMBEDDINGS)
    if torch.cuda.is_available():
        device = "cuda"
        print("Cuda is available")
    else:
        device = "cpu"

    model_row     = ModelRowSel(MODEL_ROW_STACKS).to(device)
    loss_function = torch.nn.HuberLoss().to(device) #torch.nn.BCELoss().to(device)
    optim        = torch.optim.Adam(model_row.parameters(), lr=LR_ROW_MODEL)
    running_loss = 0.0
    best_accuracy = -1.0

    for epoch in range(EPOCHS):
        linenum = 0
        model_row.train()
        with open(trainfile, 'r') as file:
            for line in file:
                data = json.loads(line) # Load the string in this line of the file
                question = data.get("question", "")
                question_tokens = tokenise_punc_sentence_with_number(question) + QUESTION_ANSWER_DELIMITER          ## List of tokens
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
                        rows_in_this_batch += ENTRY_ROW_WISE_STOP + tokenise_punc_sentence_with_number(relevant_col_entries[i])
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
                    question_tokens = tokenise_punc_sentence_with_number(question) + QUESTION_ANSWER_DELIMITER          ## List of tokens
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
                            rows_in_this_batch += ENTRY_ROW_WISE_STOP + tokenise_punc_sentence_with_number(relevant_col_entries[i])
                        
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
    return
    
def train(trainfile, valfile):
    print("Starting Training")

    if TRAIN_COL:
        train_col_model(trainfile, valfile)
    if TRAIN_ROW:
        train_row_model(trainfile, valfile)

    print("All Trainings Done!")
    cm = CM(MODEL_COL_STACKS, MODEL_ROW_STACKS)
    cm.load_individual('model_col.pth', 'model_row.pth')
    cm.save_comb('final.pth')
    print("Final models saved to : final.pth")

    return
