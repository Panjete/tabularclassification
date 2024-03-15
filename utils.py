import json
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string
import gensim
import random
import os

IMPLIES             = '<->'#"$" ## using multi-char symbol was being partitioned into individual chars by w2v
ENTRY_ROW_WISE_STOP = '|||'#"%"
IS_OF_TYPE          = ['has', 'type']

def readjson(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line)) # Load the string in this line of the file
    print(data[0])
    return data

#readjson("data/A2_val.jsonl")


def retrieve_text(filename, writefile, intm_file = "data/intm.txt"):
    ### Writes tokenised text from json file into writefile

    #tokenizer = lambda x : nltk.tokenize.word_tokenize(x)
    counts = {}
    tokenizer = lambda x: [word.lower() for word in word_tokenize(x)]# if word not in string.punctuation]
    with open(intm_file, 'w+') as wf:
        with open(filename, 'r+') as file:
            for line in file:
                data = json.loads(line) # Load the string in this line of the file

                question = data.get("question", "") #+ " " + data.get("table", {}).get("caption", "") Using caption may derail from the question and directly match some other columns that align with the caption
                question_tokens = tokenizer(question)
                question_sent = " ".join(question_tokens)
                wf.write(question_sent + "\n") 
                for token in question_tokens:
                    if token in counts.keys():
                        counts[token] += 1
                    else:
                        counts[token] = 1

                rows = data.get("table", {}).get("rows", [])
                cols = data.get("table", {}).get("cols", [])
                cols = [tokenizer(col) for col in cols]
                col_types = data.get("table", {}).get("types", [])
                col_types = [tokenizer(col_type) for col_type in col_types]
                for row in rows:
                    tokens = []
                    for i, entry in enumerate(row):
                        entry_tokens = cols[i] + [IMPLIES] + tokenizer(entry) + [ENTRY_ROW_WISE_STOP] ### Encodes what column this entry belongs to, and partitions entries
                        for token in entry_tokens:
                            if token in counts.keys():
                                counts[token] += 1
                            else:
                                counts[token] = 1
                        tokens += entry_tokens
                    row_sentence = " ".join(tokens)
                    wf.write(row_sentence + "\n")

                for col_tokenised, type_entry in zip(cols, col_types):
                    entry_tokens = col_tokenised + IS_OF_TYPE + type_entry
                    entry_sent = " ".join(entry_tokens)
                    wf.write(entry_sent + "\n")
                    for token in entry_tokens:
                        if token in counts.keys():
                            counts[token] += 1
                        else:
                            counts[token] = 1
    
    print("Interim file written")
    with open(writefile, 'w+') as wf:
        with open(intm_file, 'r+') as rf:
            for line in rf:
                words = line.split()
                for i in range(len(words)):
                    if counts[words[i]] < 2:    ## If below threshold
                        r = random.random()
                        if r < 0.2:             ## With probability = 0.2
                            words[i] = "UNK"    ## Replace with UNK token
                newline = " ".join(words)
                wf.write(newline + "\n")
    os.remove(intm_file)
    return

#retrieve_text("data/A2_val.jsonl")

def generateW2Vembeddings(corpus_file, save_file="data/w2v"):
    word2vec = gensim.models.Word2Vec(corpus_file=corpus_file, vector_size=100, workers = 4)
    word2vec.wv.save_word2vec_format(save_file, binary=False) ## Set to True later
    del word2vec
    return

def loadW2Vembeddings(save_file="data/w2v"):
    #word2vec = gensim.models.Word2Vec()
    #word2vec.wv.load_word2vec_format(save_file, binary=False) ## Set to True later
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(save_file, binary=False)
    return word_vectors



def generateFTembed(corpus_file, save_file="data/ft"):
    FasT = gensim.models.FastText(vector_size=100, workers = 4, min_count=5)
    FasT.build_vocab(corpus_file=corpus_file)
    total_words = FasT.corpus_total_words
    FasT.train(corpus_file=corpus_file, total_words=total_words, epochs=FasT.epochs)

    FasT.wv.save_word2vec_format(save_file, binary=False) ## Set to True later
    del FasT
    return

def loadFTembed(save_file="data/ft"):
    #word2vec = gensim.models.Word2Vec()
    #word2vec.wv.load_word2vec_format(save_file, binary=False) ## Set to True later
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(save_file, binary=False)
    return word_vectors

def tokenise_sentence(text):
    return [word.lower() for word in word_tokenize(text)]


def get_embeddings(word, model):
    try:
        return model.get_vector(word)
    except KeyError:
        return model.get_vector("UNK")


### Functions for test time 
def read_queries(filename):
    ## Returns a list of questions and their corresponding tables (columns for now)
    ## queries['question']   : List of List of tokens
    ## queries['table_cols'] : List of List of List of tokens  i, j -> ith question's jth columns tokens

    #tokenizer = lambda x : nltk.tokenize.word_tokenize(x)
    tokenizer = lambda x: [word.lower() for word in word_tokenize(x)]# if word not in string.punctuation]
    queries = {"questions" : [], "tables" : [], "table_cols" : [], "qids" : [], "correct_col_numbers" : []}
    numqueries = 0
    with open(filename, 'r+') as file:
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
                    break

            cols = [tokenizer(col) for col in cols]
            col_types = data.get("table", {}).get("types", [])
            col_types = [tokenizer(col_type) for col_type in col_types]
            

            table_cols = []
            for col_tokenised, type_entry in zip(cols, col_types):
                entry_tokens = col_tokenised + IS_OF_TYPE + type_entry
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

                answer_column = cols[index]                  ## The number
                answer_column_text = data_cols[answer_column] ## The text of the column name
                print("Answer column == ", answer_column)
                data_relevant_col_entries = []
                for row in data_tablerows:
                    data_relevant_col_entries.append(row[answer_column])

                answer_rows = rowss[index]
                answer_cells = []
                for row in answer_rows:
                    answer_cells.append([row, data_relevant_col_entries[row]])
        

                

                outdict = {"label_col":[answer_column_text],"label_cell":answer_cells,"label_row":answer_rows, "qid":data_qid}
                json.dump(outdict, outf)
                outf.write("\n")
                index += 1
                if index == 10:
                    return

    return

def find_col(column, cols):
    for i, col in enumerate(cols):
        if col == column:
            return i
        