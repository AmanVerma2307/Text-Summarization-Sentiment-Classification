####### Importing Libraries
import os
import torch
import numpy as np
import pandas as pd
import torch.utils
import torch.utils.data

####### Dataset
class SentimentSet(torch.utils.data.Dataset):

    def __init__(self,clean_tweets,labels,vocab, max_seq_length, pad_token, unk_token):

        ##### Variable declaration
        self.clean_tweets = clean_tweets # Cleaned tweets
        self.labels = labels # Labels
        self.pad_token, self.unk_token = pad_token, unk_token # Token demarcation
        self.ind_seq = [] # List to store indices as per Glove vocabulary
        #self.seq_len = [] # List to store sequence length

        ##### Vocabulary dictionaries
        self.word2idx = {term:idx for idx,term in enumerate(vocab)}
        self.idx2word = {idx:word for word,idx in self.word2idx.items()}

        ##### Word to index
        for idx, token_seq in enumerate(self.clean_tweets):
            self.ind_seq.append(self.tokens2indices(self.clean_tweets[idx],max_seq_length))

    def tokens2indices(self,tokens,pad_len):
        
        ##### Padding
        #missing_len = abs(pad_len - len(tokens))

        if(len(tokens) >= pad_len):
            tokens = tokens[:pad_len]
        else:
            for _ in range(pad_len - len(tokens)):
                tokens.append(self.pad_token)

        ##### Index conversion
        for idx in range(len(tokens)):

            if tokens[idx] not in self.word2idx:
                tokens[idx] = self.word2idx[self.unk_token]

            else:
                tokens[idx] = self.word2idx[tokens[idx]]

        return torch.Tensor(tokens).long()

    def __len__(self):
        return len(self.clean_tweets)
    
    def __getitem__(self, idx):
        sample = {'data':self.ind_seq[idx],'label':np.array(int(self.labels[idx]))}
        return sample