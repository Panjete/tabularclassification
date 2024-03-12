import torch
import numpy
import gensim
import pandas
import nltk
import datefinder

from utils import readjson, retrieve_text, generateW2Vembeddings

TOKENFILE = True
W2VEMBEDD = False

def train(trainfile, valfile, interim_text_file = "data/interim_text_file"): ## Change this


    ## Tokenise words and save them as sentences in interim_text_file

    if TOKENFILE:
        retrieve_text(trainfile, interim_text_file)
    print("Retrieved text, tokenized and stored in interim text file")

    ## generate embeddings
    print("Generating embeddings")
    if W2VEMBEDD:
        generateW2Vembeddings(interim_text_file, save_file="data/w2v")


    return

train("data/A2_train.jsonl", "data/A2_val.jsonl")