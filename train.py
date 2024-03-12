import numpy
import gensim
import pandas
import datefinder
import json

from utils import retrieve_text, generateW2Vembeddings, loadW2Vembeddings, tokenise_sentence

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

    ## Define Model
        
    ## Train Model
    w2v_embeddings = loadW2Vembeddings(save_file="data/w2v")
    with open(trainfile, 'r+') as file:
            for line in file:
                data = json.loads(line) # Load the string in this line of the file
                question = data.get("question", "")
                question_tokens = tokenise_sentence(question)          ## List of tokens
                cols = data.get("table", {}).get("cols", [])
                cols_tokens = [tokenise_sentence(col) for col in cols] ## List of List of tokens

                correct_col = data.get("label_col", ["NULL"])[0]       ## Uniquely One column correct

                ## Bound question length by 64
                
                ### For this point
                ### Bring CLS token closer to correct_col embedding
                ### And farther from other negative samples

    return

train("data/A2_train.jsonl", "data/A2_val.jsonl")