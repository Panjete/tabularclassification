from utils import read_queries, tokenise_sentence, loadW2Vembeddings, get_embeddings, write_outputs  
from model import Model
import torch
from torch.nn.functional import cosine_similarity

def test(testfile, predfile):
    ## Read Question and table
    queries, numqueries = read_queries(testfile) ## Reads just columns

    ## Load Models
    w2v_embeddings = loadW2Vembeddings(save_file="data/w2v")
    model_col = Model(4)
    model_col.eval()
    model_col.load_state_dict(torch.load('data/model.pth'))
    answer_columns = [] # List of numbers
    answer_rows    = [] # List of List of numbers

    for query_num in range(numqueries):
        ## Query Model1 for column
        question_tokens = queries['questions'][query_num]
        question_tensors = [torch.tensor(get_embeddings(word, w2v_embeddings)) for word in question_tokens]
        question_tensor = torch.stack(question_tensors, dim = 0) ## (Len_question * 100) tensor

        columns = queries['table_cols'][query_num]      ## List of List of tokens
        num_columns_this_query = len(columns)
        col_embs_tensors = []
        for entry in columns:                           ## List of tokens
            total_embedding = torch.tensor(get_embeddings(entry[0], w2v_embeddings))
            for word in entry[1:]:
                total_embedding += torch.tensor(get_embeddings(word, w2v_embeddings))
        
            col_embs_tensors.append(total_embedding)       ## Embedding for this column
        col_embs = torch.stack(col_embs_tensors, dim = 0)  ## (#columns * 100) tensor embedding, for all columns

        CLS = model_col(question_tensor)                           ## (1 * 100) tensor
        CLS_broadcast = CLS.repeat(num_columns_this_query, 1)      ## (Len_question * 100) tensor
        similarities = cosine_similarity(CLS_broadcast, col_embs)  ## (Len_question, 1) tensor
        
        max_similarity_column = torch.argmax(similarities, dim=0).item()
        answer_columns.append(max_similarity_column)
        answer_rows.append([0])
        ## Read in appropriate columns
                        
        ## Query Model 2 for row
        if query_num == 13:
            break

    ## Write to output
    write_outputs(testfile, predfile, answer_columns, answer_rows)
    
    return

test("data/A2_val.jsonl","data/G_val.jsonl" )