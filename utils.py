import json
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string
import gensim
import random
import os

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

                question = data.get("question", "")
                question_tokens = tokenizer(question)
                question_sent = " ".join(question_tokens)
                wf.write(question_sent + "\n") 
                for token in question_tokens:
                    if token in counts.keys():
                        counts[token] += 1
                    else:
                        counts[token] = 1

                rows = data.get("table", {}).get("rows", [])
                for row in rows:
                    for entry in row:
                        entry_tokens = tokenizer(entry)
                        entry_sent = " ".join(entry_tokens)
                        wf.write(entry_sent + "\n")
                        for token in entry_tokens:
                            if token in counts.keys():
                                counts[token] += 1
                            else:
                                counts[token] = 1

                cols = data.get("table", {}).get("cols", [])
                for entry in cols:
                    entry_tokens = tokenizer(entry)
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
                    if counts[words[i]] < 3:    ## If below threshols
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
    return