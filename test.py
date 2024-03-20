from utils import tokenise_punc_sentence,\
      get_embeddings, write_outputs, pad_tokens, tokenise_punc_sentence_with_number, tokenise_punc_sentence_question
import json
import torch
from combModels       import CM
import gensim.downloader 
from configs import *

def test(testfile, predfile):

    embeddings = gensim.downloader.load(EMBEDDINGS)
    print("Starting testing!")
    if torch.cuda.is_available():
        device = "cuda"
        print("Cuda is available")
    else:
        device = "cpu"
    
    cm = CM(MODEL_COL_STACKS, MODEL_ROW_STACKS).to(device)
    cm.load_combined('final.pth')

    model_col = cm.model_col
    model_col.eval()
   
    model_row = cm.model_row
    model_row.eval()
    answer_columns = [] # List of numbers
    answer_rows    = [] # List of List of numbers


    with open(testfile, 'r') as file:
        for line in file:
            data = json.loads(line) # Load the string in this line of the file
            question = data.get("question", "")
            question_tokens = tokenise_punc_sentence(question) + QUESTION_ANSWER_DELIMITER          ## List of tokens
            cols = data.get("table", {}).get("cols", [])
            columns = [tokenise_punc_sentence(col) for col in cols]
            
            final_tensor = []
            max_init = 0
            for col_tokens in columns:
                max_init = max(max_init, len(col_tokens) + 1 + len(question_tokens))
            #print("Max Init for this question = ", max_init)
            for col_tokens in columns:
                this_column_tokens  = pad_tokens(question_tokens + QUESTION_ANSWER_DELIMITER + col_tokens, max_init)
                #print("TOKENS FOR THIS QUESTION VAL = ", this_column_tokens)
                token_tensors    = [torch.tensor(get_embeddings(word, embeddings), device=device, requires_grad=False, dtype=torch.float32) for word in this_column_tokens]
                col_tensor       = torch.stack(token_tensors, dim = 0)                  ## (Len_question+col, 100) tensor
                final_tensor.append(col_tensor)

            final_tensor = torch.stack(final_tensor, dim = 0)
            likelihood_of_cols = model_col(final_tensor)
            #print("Predicted tensors and argmax = ", likelihood_of_cols, torch.argmax(likelihood_of_cols))
            max_similarity_column = torch.argmax(likelihood_of_cols)

            ## NOW COMPUTE ROW

            question = data.get("question", "")
            question_tokens = tokenise_punc_sentence_question(question)          ## List of tokens
            cols = data.get("table", {}).get("cols", [])
            cols_cleaned = [tokenise_punc_sentence_question(col) for col in cols]
            cols_cleaned = [col[:min(2, len(col))] for col in cols_cleaned]
            #numcols = len(cols)
            
            rows = data.get("table", {}).get("rows", [])

            max_row_score = -1.0
            max_row_at    = -1
            for row_index, row in enumerate(rows):
                this_row_tokens   = question_tokens + tokenise_punc_sentence_with_number(row, True, cols_cleaned) #+ ROW_ANSWER_DELIMITER + tokenise_punc_sentence_with_number([cell_for_this_row], False)
                token_tensors    = [torch.tensor(get_embeddings(word, embeddings), device=device, requires_grad=False, dtype=torch.float32) for word in this_row_tokens]
                row_tensor       = torch.stack(token_tensors, dim = 0)                  ## (Len_question+col, 100) tensor
                likelihood_of_row = model_row(row_tensor).item()

                if likelihood_of_row > max_row_score:
                    max_row_score = likelihood_of_row
                    max_row_at    = row_index
            

            
            '''
            rows  = data.get("table", {}).get("rows", [])
            num_rows     = len(rows)

            relevant_col_entries = [row[max_similarity_column] for row in rows]

            num_batches_rows = (num_rows//32) + 1
            row_predictions  = []
            for row_batch in range(num_batches_rows):
                start_index = 32 * row_batch
                end_index = min(start_index + 32, num_rows) ## Both not to be included
                rows_in_this_batch = question_tokens
                for i in range(start_index, end_index):
                    rows_in_this_batch += ENTRY_ROW_WISE_STOP + tokenise_punc_sentence_with_number(relevant_col_entries[i])
                padded_row = pad_tokens(rows_in_this_batch, ROW_TOKENS)
                tokenised_rows_batch_tensors= [torch.tensor(get_embeddings(token, embeddings), requires_grad=False, dtype=torch.float32, device=device) for token in padded_row]
                batch_prepared = torch.stack(tokenised_rows_batch_tensors, dim = 0)

                predicted_tensor = model_row(batch_prepared)
                prediction = round(torch.clip(predicted_tensor, -20.0, 31.0).item())
                if prediction > -1 and prediction < 32:
                    row_predictions.append(prediction + start_index)

            if len(row_predictions) == 0:
                           row_predictions = [0]
            '''

                    
            answer_columns.append(max_similarity_column)
            answer_rows.append([max_row_at])
        ## Read in appropriate columns
                        
        ## Query Model 2 for row

    ## Write to output
    write_outputs(testfile, predfile, answer_columns, answer_rows)
    return

#test("data/A2_val.jsonl","data/G_val.jsonl" )

## Use python3 evaluate_submission.py data/A2_val_prediction.jsonl data/G_val.jsonl