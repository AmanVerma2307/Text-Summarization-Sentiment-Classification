###### Importing Libraries
import os
import torch
import numpy as np
import pandas as pd
from Preprocess import preprocess
from DataLoader import SentimentSet

###### Preprocessing
X,y = preprocess('./train.csv')

####### Vocabulary
vocab,embeddings = [],[]
with open('./glove.twitter.27B/glove.twitter.27B.100d.txt','rt',encoding="utf8") as fi:
    full_content = fi.read().strip().split('\n')
for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    vocab.append(i_word)

vocab = np.array(vocab)

vocab = np.insert(vocab, 0, '<pad>')
vocab = np.insert(vocab, 1, '<unk>')

###### Dataloader
SentimentDataset = SentimentSet(X,y,vocab,10,"<pad>","<unk>")
#test_idx = 30
#print(SentimentDataset.__getitem__(test_idx)['data'])

train_loader = torch.utils.data.DataLoader(SentimentDataset,
                                           batch_size=32,
                                           shuffle=True,
                                           drop_last=False)

for item in iter(train_loader):
    print('============================')
    print(item['data'].shape, item['label'].shape)