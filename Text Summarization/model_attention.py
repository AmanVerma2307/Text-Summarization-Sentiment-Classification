####### Importing Libraries
import os
import gc
import random
import torch
import numpy as np
import pandas as pd
import torch.nn.functional
from Preprocess import preprocess
from VocabularyGeneration import vocab_gen
from GloVe_Layer import Glove_Gen
from EmbeddingLayer import Embedding_Layer
#print('Importing done')

###### Seq2Seq Model
device = torch.device("cuda:0")
X, y = preprocess('./Reviews_q2_main.csv')
word2idx, idx2word = vocab_gen(X+y,"<sos>","<eos>")
#embedding_layer = Glove_Gen(word2idx,"<sos>","<eos>")
del(X,y)
gc.collect()
#print('Preprocessing Done')

##### Encoder
class EncoderRNN(torch.nn.Module):

    def __init__(self,embedding_dim,hidden_dim,num_layers):

        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        #### Embedding layer
        self.embedding = Glove_Gen(word2idx,"<sos>","<eos>")
        #self.embedding = Embedding_Layer()

        #### BiLSTM layer
        self.LSTM = torch.nn.LSTM(self.embedding_dim,
                                  self.hidden_dim,
                                  self.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        
    def forward(self, input):
        input = self.embedding(input)
        ouptut, hidden = self.LSTM(input)
        return ouptut, hidden
    
##### Attention Module
class BahdanauAttention(torch.nn.Module):

    def __init__(self,hidden_dim):
        super().__init__()
        self.Wq = torch.nn.Linear(hidden_dim, hidden_dim)
        self.Wk = torch.nn.Linear(2*hidden_dim, hidden_dim)
        self.Va = torch.nn.Linear(hidden_dim, 1)

    def forward(self, query, keys):
        # query: [N,1,D], keys: [N,T,D]
        scores = self.Va(torch.tanh(self.Wq(query)+self.Wk(keys))) # scores -> [N,T,1]  
        scores = scores.squeeze(2).unsqueeze(1) # scores -> [N,1,T]
        weights = torch.nn.functional.softmax(scores, dim=-1) # scores -> [N,1,T]: Softmax
        context = torch.bmm(weights, keys) # context -> [N,1,D]: Weighted summation
        return context, weights

##### Decoder
class AttnDecoderRNN(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers, op_dim, max_len):

        #### Defining essentials
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.op_dim = op_dim
        self.max_len = max_len

        #### Embedding layer
        self.embedding = Glove_Gen(word2idx,"<sos>","<eos>")
        #self.embedding = Embedding_Layer()

        #### Attention layer
        self.attention = BahdanauAttention(self.hidden_dim)

        #### LSTM layer
        self.LSTM = torch.nn.LSTM(self.embedding_dim+2*self.hidden_dim,
                                  self.hidden_dim,
                                  self.num_layers,
                                  batch_first=True,
                                  bidirectional=False)
        
        #### Linear layer
        self.op_layer = torch.nn.Linear(self.hidden_dim,self.op_dim)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward_step(self, input, hidden, encoder_outputs):
        output = self.embedding(input) # Embedding layer: output -> [N,1,D]
        query = hidden[0].permute(1,0,2) # query: [1,N,D] -> [N,1,D]
        context, _ = self.attention(query,encoder_outputs) # context: [N,1,D]
        output = torch.cat((output.float(),context),dim=2) # output -> [N,1,E+2D]
        output, hidden = self.LSTM(output, hidden) # output -> [N,1,D]
        output = self.op_layer(output) # output -> [N,D]
        output = self.softmax(output) # output -> [N,D]
        return output, hidden
    
    def forward(self, encoder_outputs, encoder_hidden, target_tensor, tf_prob):

        batch_size = encoder_hidden[0].size(1) # Batch size
        decoder_input = torch.zeros(batch_size,1).long().to(device) # "<sos>" token
        decoder_hidden = encoder_hidden
        decoder_outputs = [] # List to store ouptuts

        for i in range(self.max_len):

            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)

            if random.random() < tf_prob: # Teacher forcing
                decoder_input = target_tensor[:,i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs,dim=1)
        return decoder_outputs, decoder_hidden
    
##### Testing
#device = torch.device("cuda:0")
#attenion_module = BahdanauAttention(75).to(device)
#query = torch.randn(32,1,75*2).to(device)
#keys = torch.randn(32,10,75).to(device)
#context, weights = attenion_module.forward(query,keys)

#device = torch.device("cuda:0")
#encoder = EncoderRNN(100,75,1).to(device)
#decoder = AttnDecoderRNN(100,75,1,26952,10).to(device)

#output, hidden = encoder.forward((torch.zeros(32,10).long()).to(device))
#print(output.shape, hidden[0].shape, hidden[1].shape)
#decoder_outputs, decoder_hidden = decoder.forward(output,(torch.sum(hidden[0],axis=0,keepdim=True), 
#                                                    torch.sum(hidden[1],axis=0,keepdim=True)), 
#                                                  (torch.zeros(32,10).long()).to(device),
#                                                  tf_prob=0.5)
#print(decoder_outputs.shape, decoder_hidden[0].shape, decoder_hidden[1].shape)
