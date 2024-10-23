####### Importing Libraries
import os
import gc
import random
import torch
import numpy as np
import pandas as pd
from Preprocess import preprocess
from VocabularyGeneration import vocab_gen
from GloVe_Layer import Glove_Gen
from EmbeddingLayer import Embedding_Layer

###### Seq2Seq Model
device = torch.device("cuda:0")
X, y = preprocess('./Reviews_q2_main.csv')
word2idx, idx2word = vocab_gen(X+y,"<sos>","<eos>")
#embedding_layer = Glove_Gen(word2idx,"<sos>","<eos>")
del(X,y)
gc.collect()

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
    
##### Decoder
class DecoderRNN(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers, op_dim, max_len):
        
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.op_dim = op_dim
        self.max_len = max_len

        #### Embedding layer
        self.embedding = Glove_Gen(word2idx,"<sos>","<eos>")
        #self.embedding = Embedding_Layer()

        #### BiLSTM layer
        self.LSTM = torch.nn.LSTM(self.embedding_dim,
                                  self.hidden_dim,
                                  self.num_layers,
                                  batch_first=True,
                                  bidirectional=False)
        
        #### Linear layer
        self.op_layer = torch.nn.Linear(self.hidden_dim,self.op_dim)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = torch.nn.functional.relu(output.float())
        output, hidden = self.LSTM(output, hidden)
        output = self.op_layer(output)
        output = self.softmax(output)
        return output, hidden

    def forward(self, encoder_hidden, target_tensor, tf_prob):

        #batch_size = encoder_outputs.size(0) # Batch size
        batch_size = encoder_hidden[0].size(1) # Batch size
        decoder_input = torch.zeros(batch_size,1).long().to(device) # "<sos>" token
        decoder_hidden = encoder_hidden
        decoder_outputs = [] # List to store ouptuts
        #target_tensor = target_tensor.to(device)

        for i in range(self.max_len):

            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
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
#encoder = EncoderRNN(100,75,1).to(device)
#output, (h, c) = encoder.forward((torch.zeros(32,10).long()).to(device))
#print(output.shape, h.shape, c.shape)

#device = torch.device("cuda:0")
#encoder = EncoderRNN(100,75,1).to(device)
#decoder = DecoderRNN(100,75,1,21352,10).to(device)

#output, hidden = encoder.forward((torch.zeros(32,100).long()).to(device))
#print(output.shape, hidden[0].shape, hidden[1].shape)
#decoder_outputs, decoder_hidden = decoder.forward((torch.sum(hidden[0],axis=0,keepdim=True), 
#                                                        torch.sum(hidden[1],axis=0,keepdim=True)), 
#                                                  (torch.zeros(32,10).long()).to(device),
#                                                  tf_prob=0.5)
#print(output.shape, hidden[0].shape, hidden[1].shape)
#print(decoder_outputs.shape)