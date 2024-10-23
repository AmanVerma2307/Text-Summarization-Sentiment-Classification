###### Importing Libraries
import os
import torch
import numpy as np
import pandas as pd

###### Glove embedding layer

##### Glove Embeddings
def GloVe():
    
    embeddings = []
    with open('./glove.twitter.27B/glove.twitter.27B.100d.txt','rt',encoding="utf8") as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        embeddings.append(i_embeddings)

    embeddings = np.array(embeddings)

    pad_emb = np.zeros((1,embeddings.shape[1]))   #embedding for '<pad>' token.
    unk_emb = np.mean(embeddings,axis=0,keepdims=True)    #embedding for '<unk>' token.

    embeddings = np.vstack((pad_emb,unk_emb,embeddings))

    ###### Fixing values in embedding module
    embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float())
    return embedding_layer

###### Test
device = torch.device("cuda:0")
embedding_layer = GloVe()
embedding_layer = embedding_layer.to(device)
a = embedding_layer((torch.zeros(64,100).long()).to(device))
print(a.shape)