import json
from nltk.tokenize import WordPunctTokenizer
from unidecode import unidecode
from configs import *
import os
import gensim

# All utility code housed here. Helper functions for writing/reading/tokenising/loading emmbeddings etc.
PunctTokenizerObject = WordPunctTokenizer()
stop_words           = set(['the', '.', '"', '\'', '!', 'as', 'a', 'an', '(', ')'] + ['i', 'me', 'my', 'myself', 'we', 'our', 'ours'])

def readjson(filename):
    ## real json file, line by line
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line)) # Load the string in this line of the file
    return data

### Tokenisers ###

def tokenise_punc_sentence(text):
    ## Take in a string, tokenise and normalise it. Replace numbers with <NUM> token
    tokens = PunctTokenizerObject.tokenize(text)

    tokens = [unidecode(token) for token in tokens]
    tokens = [word.lower() for word in tokens]
    tokens = [token for token in tokens if token not in stop_words]
    for i in range(len(tokens)):
        if tokens[i].isdigit():
            tokens[i] = "NUM"
    
    return tokens

def tokenise_punc_sentence_question(text):
    ## Take in a string, tokenise and normalise it.
    tokens = PunctTokenizerObject.tokenize(text)
    tokens = [unidecode(token) for token in tokens]
    tokens = [word.lower() for word in tokens]
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def tokenise_punc_sentence_with_number(text, flag, cols = []):
    ## converts list of tokens to reasonable sized list of parsed, normalised tokens
    ## Flag == true means this is a row

    if flag:
        # Row, alreasy a list of entries
        tokens = [PunctTokenizerObject.tokenize(entry) for entry in text] ## List of (List of tokens) -> per column entry, max cap 64
        total_tokens = sum([len(token_el) for token_el in tokens])

        if total_tokens > MAX_TOKENS_ROW and flag:
            tokens = [token_el[:min(1, len(token_el))] for token_el in tokens]

        tokens = [[unidecode(token) for token in token_el] for token_el in tokens]
        tokens = [[word.lower() for word in token_el] for token_el in tokens]
        tokens = [[token for token in token_el if token not in stop_words] for token_el in tokens]
        tokens = [col_ann + COL_NAME_COL_ENTRY_DELIMITER + token_el + ENTRY_ROW_WISE_STOP for col_ann, token_el in zip(cols, tokens)]
        final_chain = []
        for token in tokens:
            final_chain += token
        return final_chain
    else:
        # Entry
        text   = " ".join(text)
        tokens = PunctTokenizerObject.tokenize(text)
        tokens = [unidecode(token) for token in tokens]
        tokens = [word.lower() for word in tokens]
        tokens = [token for token in tokens if token not in stop_words]
        if len(tokens) > MAX_TOKENS_ENTRY:
            tokens = tokens[:MAX_TOKENS_ENTRY]
        return tokens

    
def get_embeddings(word, model):
    ## Query the word embedding model for embeddings of a particular word
    try:
        return model[word]
    except:
        return model["pad"]


### Functions for test time ###
def read_queries(filename):
    ## Returns a list of questions and their corresponding tables (columns for now)
    ## queries['question']   : List of List of tokens
    ## queries['table_cols'] : List of List of List of tokens  i, j -> ith question's jth columns tokens
    #tokenizer = lambda x : nltk.tokenize.word_tokenize(x)
    tokenizer = lambda x : tokenise_punc_sentence(x)
    #tokenizer = lambda x: [word.lower() for word in word_tokenize(x)]# if word not in string.punctuation]
    queries = {"questions" : [], "tables" : [], "table_cols" : [], "qids" : [], "correct_col_numbers" : []}
    numqueries = 0
    with open(filename, 'r') as file:
        for line in file:
            numqueries += 1
            data = json.loads(line) # Load the string in this line of the file

            qid = data.get("qid", "") 
            question = data.get("question", "") 
            question_tokens = tokenizer(question)
            queries["qids"].append(qid)
            queries["questions"].append(question_tokens)

            cols = data.get("table", {}).get("cols", [])
            correct_col = data.get("label_col", ["NULL"])[0]
            correct_col_number = -1
            for num, col in enumerate(cols):
                if col == correct_col:
                    correct_col_number = num
                    #print("Correct column number = ", correct_col_number)
                    break

            cols = [tokenizer(col) for col in cols]
            col_types = data.get("table", {}).get("types", [])
            col_types = [tokenizer(col_type) for col_type in col_types]
            

            table_cols = []
            for col_tokenised, type_entry in zip(cols, col_types):
                entry_tokens = col_tokenised #+ IS_OF_TYPE + type_entry
                table_cols.append(entry_tokens)

            queries["table_cols"].append(table_cols)
            queries["correct_col_numbers"].append(correct_col_number)
    
    return queries, numqueries

def write_outputs(inputfile, outputfile, cols, rowss):
    ## cols is a list of numbers -> ith represents the correct column number for the ith query
    ## rowss is a list of list of numbers -> ith represents the correct rows for the ith query
    index = 0
    with open(outputfile, 'w') as outf:
        with open(inputfile, 'r') as inf:
            for line in inf:
                data = json.loads(line) # Load the string in this line of the file
                data_cols = data.get("table", {}).get("cols", [])
                data_qid = data.get("qid", "") 
                data_tablerows = data.get("table", {}).get("rows", [])

                num_cols = len(data_cols)
                num_rows = len(data_tablerows)

                #print("original proposition columns = ", )
                
                answer_column = cols[index]                  ## The number
                answer_column = max(0, min(num_cols-1, answer_column))
        
                answer_column_text = data_cols[answer_column] ## The text of the column name
                #print("Answer column == ", answer_column)
                data_relevant_col_entries = []
                for row in data_tablerows:
                    data_relevant_col_entries.append(row[answer_column])

                answer_rows = rowss[index]
                answer_cells = []
                for row in answer_rows: ## Will always be 1 in my assumption
                    row = max(0, min(num_rows-1, row))
                    answer_cells.append([row, answer_column_text])
        

                outdict = {"label_col":[answer_column_text],"label_cell":answer_cells,"label_row":answer_rows, "qid":data_qid}
                json.dump(outdict, outf)
                outf.write("\n")
                index += 1

    return

def find_col(column, cols):
    ## Finds index of column from a list of columns
    for i, col in enumerate(cols):
        if col == column:
            return i
        
def pad_tokens(tokens, desired_length=20):
    ## Will convert a list of tokens to desired length
    curlen = len(tokens)
    if(len(tokens) >= desired_length):
        tokens = tokens[:desired_length]
    else:
        tokens += ["pad" for _ in range(desired_length-curlen)]
    return tokens
  

def retrieve_text(filename, writefile, intm_file = "data/intm.txt"):
    ### Writes tokenised text from json file into writefile
    ### To be used if to create a corpus of text encoding relevant information, later used for generating word embeddings
    tokenizer = lambda x : tokenise_punc_sentence(x)
    with open(intm_file, 'w') as wf:
        with open(filename, 'r') as file:
            for line in file:
                data = json.loads(line) # Load the string in this line of the file

                question = data.get("question", "") #+ " " + data.get("table", {}).get("caption", "") Using caption may derail from the question and directly match some other columns that align with the caption
                question_tokens = tokenizer(question)
                question_sent = " ".join(question_tokens)
                wf.write(question_sent + "\n") 
                correct_col        = data.get("label_col", ["NULL"])[0]
                correct_col_tokens = tokenizer(correct_col)
                correct_col_sent   = " ".join(question_tokens + [QUESTION_ANSWER_DELIMITER] + correct_col_tokens)
                wf.write(correct_col_sent + "\n") 

                rows = data.get("table", {}).get("rows", []) ## List of List of sentences
                cols = data.get("table", {}).get("cols", [])
                cols = [tokenizer(col) for col in cols]
                col_types = data.get("table", {}).get("types", [])
                col_types = [tokenizer(col_type) for col_type in col_types]

                ## Writing out per-column, along with the element wise column name
                for i in range(len(cols)):
                    tokens = []
                    cols_i = cols[i]
                    for row in rows:
                        tokens += cols_i + [IMPLIES] + tokenizer(row[i]) + [ENTRY_ROW_WISE_STOP]
                    col_sentence = " ".join(tokens)
                    wf.write(col_sentence + "\n")

                ## Writing out per-row, along with the element wise column name
                for row in rows:
                    tokens = []
                    for i, entry in enumerate(row):
                        entry_tokens = cols[i] + [IMPLIES] + tokenizer(entry) + [ENTRY_ROW_WISE_STOP] ### Encodes what column this entry belongs to, and partitions entries
                        tokens += entry_tokens
                    row_sentence = " ".join(tokens)
                    wf.write(row_sentence + "\n")

                for col_tokenised, type_entry in zip(cols, col_types):
                    entry_tokens = col_tokenised + IS_OF_TYPE + type_entry
                    entry_sent = " ".join(entry_tokens)
                    wf.write(entry_sent + "\n")
    
    print("Interim file written")
    with open(writefile, 'w') as wf:
        with open(intm_file, 'r') as rf:
            for line in rf:
                words = line.split()
                newline = " ".join(words)
                wf.write(newline + "\n")
    os.remove(intm_file)
    return


### Generating (from corpus text file) and loading word embeddings functions ###

def generateW2Vembeddings(corpus_file, save_file="data/w2v"):
    word2vec = gensim.models.Word2Vec(corpus_file=corpus_file, vector_size=100, workers = 4)
    word2vec.wv.save_word2vec_format(save_file, binary=False) ## Set to True later
    del word2vec
    return

def loadW2Vembeddings(save_file="data/w2v"):
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(save_file, binary=False)
    return word_vectors



def generateFTembed(corpus_file, save_file="data/ft"):
    FasT = gensim.models.FastText(vector_size=100, workers = 4, min_count=5)
    FasT.build_vocab(corpus_file=corpus_file)
    total_words = FasT.corpus_total_words
    FasT.train(corpus_file=corpus_file, total_words=total_words, epochs=FasT.epochs)
    gensim.models.fasttext.save_facebook_model(FasT, save_file)
    return

def loadFTembed(save_file="data/ft"):
    FasT = gensim.models.fasttext.load_facebook_model(save_file) 
    return FasT
