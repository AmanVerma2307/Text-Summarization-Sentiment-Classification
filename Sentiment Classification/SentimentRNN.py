###### Importing Libraries
import os
import torch
import numpy as np
import pandas as pd
from GloveEmbeddingLayer import GloVe

###### Defining essentials
vocab_size = 1193516
embedding_dim = 100

###### Sentiment RNN
class SentimentRNN(torch.nn.Module):

    """
    RNN Module for sentiment analysis
    """

    def __init__(self,vocab_size,embedding_dim,hidden_dim,n_layers,num_class):

        ##### Defining Essentials
        super().__init__()
        self.num_class = num_class # Number of output class
        self.embedding_dim = embedding_dim # Embedding dimensions
        self.hidden_dim = hidden_dim # Hidden dimensions
        self.num_layers = n_layers # Number of layers
        self.vocab_size = vocab_size # Vocabulary size

        ##### Defining Layers
        self.embedding_layer = GloVe() # GloVe layer
        self.RNN = torch.nn.RNN(self.embedding_dim,self.hidden_dim,
                                self.num_layers,batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_dim,self.num_class)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, hidden):
        x = x.long() 
        embeddings = self.embedding_layer(x)
        rnn_output, hidden = self.RNN(embeddings,hidden)
        rnn_out = rnn_output[:,-1,:]
        rnn_out = self.fc(rnn_out)
        #rnn_out = self.softmax(rnn_out)
        return rnn_out
    
    def init_hidden(self,batch_size):
        return torch.zeros(self.num_layers,batch_size,self.hidden_dim).float()

###### Testing
#device = torch.device("cuda:0")    
#rnn_model = SentimentRNN(vocab_size,embedding_dim,75,1,3).to(device)
#input_tensor = torch.from_numpy(np.random.randint(0,10000,size=(8,100))).long()
#op = rnn_model(input_tensor.to(device),rnn_model.init_hidden(8).to(device))
#print(op.shape)