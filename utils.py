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
    queries = {"questions" : [], "tables" : [], "table_cols" : [], "qid" : []}
    numqueries = 0
    with open(filename, 'r+') as file:
        for line in file:
            numqueries += 1
            data = json.loads(line) # Load the string in this line of the file

            qid = data.get("qid", "") 
            question = data.get("question", "") 
            question_tokens = tokenizer(question)
            queries["qid"].append(qid)
            queries["questions"].append([question_tokens])

            cols = data.get("table", {}).get("cols", [])
            cols = [tokenizer(col) for col in cols]
            col_types = data.get("table", {}).get("types", [])
            col_types = [tokenizer(col_type) for col_type in col_types]

            table_cols = []
            for col_tokenised, type_entry in zip(cols, col_types):
                entry_tokens = col_tokenised + IS_OF_TYPE + type_entry
                table_cols.append([entry_tokens])

            queries["table_cols"].append([table_cols])
    
    return queries, numqueries