####### Importing Libraries
import os
import torch
import numpy as np
import pandas as pd
import torch.utils
import torch.utils.data
from Preprocess import preprocess
from VocabularyGeneration import vocab_gen

####### DataLoader
class SummarySet(torch.utils.data.Dataset):

    def __init__(self,X,y,word2idx,idx2word,max_len,sos_token,eos_token):
        
        self.word2idx = word2idx # Vocabulary
        self.idx2word = idx2word # Inverse vocabulary
        self.X_ind = [] # List to store indices of the input
        self.y_ind = [] # List to store indices of the output
        self.max_len = max_len # Maximum length
        self.sos_token = sos_token
        self.eos_token = eos_token

        for x_sent in X:
            self.X_ind.append(self.token2indices(x_sent.split()))

        for y_sent in y:
            self.y_ind.append(self.token2indices(y_sent.split()))
        
    def token2indices(self,tokens):

        token_idx = []
        tokens.append(self.eos_token) # <EOS> token

        if(len(tokens) >= self.max_len):
            tokens = tokens[:self.max_len]
        else:
            for _ in range(self.max_len - len(tokens)):
                tokens.append(self.sos_token)

        for token in tokens:
            token_idx.append(self.word2idx[token])
        #for token in tokens:
        #    if(self.word2idx[token] <= 21351):
        #        token_idx.append(self.word2idx[token])
        #    else:
        #        token_idx.append(0)

        return torch.Tensor(token_idx).long()
    
    def __len__(self):
        return len(self.X_ind)
    
    def __getitem__(self,idx):
        sample = {'data':self.X_ind[idx],'label':self.y_ind[idx]}
        return sample

####### Testing
#X, y = preprocess('./Reviews_q2_main.csv')
#word2idx, idx2word = vocab_gen(X+y,"<sos>","<eos>")
#Dataset = SummarySet(X,y,word2idx,idx2word,30,"<sos>","<eos>")
#print(Dataset.__getitem__(100))

#Dataloader = torch.utils.data.DataLoader(Dataset,
#                                         batch_size=32,
#                                         shuffle=True,
#                                         drop_last=False)

#for item in iter(Dataloader):
#    print(item['data'].shape,item['label'].shape)
