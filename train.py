import numpy
import gensim
import pandas
import datefinder
import json
import torch
import random

from utils import retrieve_text, generateW2Vembeddings, loadW2Vembeddings, tokenise_sentence, get_embeddings
from model import Model

TOKENFILE = False
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
    print("Generated embeddings")
    ## Define Model, Loss and Optimiser
    model = Model(2)
    loss_function = torch.nn.CosineEmbeddingLoss()
    optim = torch.optim.Adam(model.parameters())
    print("Defined Model, Loss and Optim")
        
    ## Train Model
    linenum = 0
    w2v_embeddings = loadW2Vembeddings(save_file="data/w2v")
    with open(trainfile, 'r+') as file:
            for line in file:
                data = json.loads(line) # Load the string in this line of the file
                question = data.get("question", "")
                question_tokens = tokenise_sentence(question)          ## List of tokens
                cols = data.get("table", {}).get("cols", [])
                correct_col = data.get("label_col", ["NULL"])[0]      ## Uniquely One column correct

                training_embs = [correct_col]
                while(len(training_embs)< 6):
                     rand_int = random.randint(0, len(cols)-1)
                     if cols[rand_int] != correct_col:
                          training_embs.append(cols[rand_int])

                #print("Question = ", question, " Training embs = ", training_embs)
                training_embs = [tokenise_sentence(col) for col in training_embs] ## List of List of tokens
               
                training_embs_tensors = []
                for entry in training_embs:
                    total_embedding = torch.tensor(get_embeddings(entry[0], w2v_embeddings))
                    for word in entry[1:]:
                        total_embedding += torch.tensor(get_embeddings(word, w2v_embeddings))
                
                    training_embs_tensors.append(total_embedding)
                training_embs_tensor = torch.stack(training_embs_tensors, dim = 0)  # 6 * 100 tensor
                #print("Train Embs Tensor shape = ", training_embs_tensor.shape)
                labels = [1, -1, -1, -1, -1, -1]
                labels = torch.tensor(labels)

                ## Bound question length by 20
                # if(len(question_tokens) >= 20):
                #      question_tokens = question_tokens[:20]
                # else:
                #      question_tokens += ["PADDING" for _ in range(20-len(question_tokens))]
                
                question_tensors = [torch.tensor(get_embeddings(word, w2v_embeddings)) for word in question_tokens]
                question_tensor = torch.stack(question_tensors, dim = 0)
                #print("question Tensor shape = ", question_tensor.shape)
                ### Bring CLS token closer to correct_col embedding
                ### And farther from other negative samples
                CLS = model(question_tensor)
                #print("Model Output shape = ", CLS.shape)
                CLS_broadcasted = CLS.repeat(6, 1)
                #print("Model Output Broadcased shape = ", CLS_broadcasted.shape)

                loss = loss_function(CLS_broadcasted, training_embs_tensor, labels)
                loss.backward()
                optim.step()
                linenum += 1
                if linenum%1000 == 0:
                     print("loss in iter {} = {}".format(linenum, loss.item()))

                
    torch.save(model.state_dict(), 'data/model.pth')
    return

train("data/A2_train.jsonl", "data/A2_val.jsonl")