####### Importing Libraries
import os
import gc
import time
import random
import argparse
import tqdm
import torch
import numpy as np
import pandas as pd
from Preprocess import preprocess
from VocabularyGeneration import vocab_gen
from DataLoader import SummarySet
from model import EncoderRNN, DecoderRNN
from model_attention import AttnDecoderRNN

####### Input arguments
parser = argparse.ArgumentParser()

parser.add_argument('--max_len',
                    type=int,
                    help="Maximum length")
parser.add_argument('--tf_prob',
                    type=float,
                    help="Teacher forcing probability")
parser.add_argument('--model',
                    type=str,
                    help="Choice of model: lstm or lstm_attn")
parser.add_argument('--model_name',
                    type=str,
                    help="Name of the model to be saved")

args = parser.parse_args()
print('Importing and parsing done')

####### Data Processing

###### Vocabulary generation
X, y = preprocess('./Reviews_q2_main.csv')
word2idx, idx2word = vocab_gen(X+y,"<sos>","<eos>")
del(X,y)
gc.collect()
print('Vocabulary generated')

###### Dataloader
##### Dataset
X_train, y_train = preprocess('./train.csv')
X_val, y_val = preprocess('./val.csv')

TrainDataset = SummarySet(X_train,y_train,word2idx,
                          idx2word,args.max_len,"<sos>","<eos>")
del(X_train)
gc.collect()


ValDataset = SummarySet(X_val,y_val,word2idx,
                          idx2word,args.max_len,"<sos>","<eos>")
del(X_val)
gc.collect()
 
##### Dataloader
TrainLoader = torch.utils.data.DataLoader(TrainDataset,
                                   batch_size=160,
                                   shuffle=True,
                                   drop_last=False)
ValLoader = torch.utils.data.DataLoader(ValDataset,
                                   batch_size=160, 
                                   shuffle=False,
                                   drop_last=False)
print('Dataloader set')

####### Model training
###### Defining essentials
embedding_dim = 100
hidden_dim = 75
sos_token = '<sos>'
eos_token = '<eos>'
vocab_size = 26952
train_total = 20000
val_total = 3750
enc_num_layers = 1
dec_num_layers = 1
device = torch.device("cuda:0")
train_loss = []
val_loss = []

###### Model
class lstm_model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        #self.tf_prob = tf_prob # Teacher forcing probability
        self.encoder = EncoderRNN(embedding_dim,hidden_dim,enc_num_layers) # Encoder layer
        self.decoder = DecoderRNN(embedding_dim,hidden_dim,dec_num_layers,vocab_size,args.max_len) # Decoder layer

    def forward(self,source,target,tf_prob):
        encoder_output, encoder_hidden = self.encoder.forward(source)
        decoder_outputs, decoder_hidden = self.decoder.forward((torch.sum(encoder_hidden[0],axis=0,keepdim=True),
                                                                torch.sum(encoder_hidden[1],axis=0,keepdim=True)),
                                                                target.to(device),tf_prob)
        return decoder_outputs, decoder_hidden

class lstm_attn_model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = EncoderRNN(embedding_dim,hidden_dim,enc_num_layers) # Encoder layer
        self.decoder = AttnDecoderRNN(embedding_dim,hidden_dim,dec_num_layers,vocab_size,args.max_len) # Decoder layer

    def forward(self,source,target,tf_prob):
        encoder_output, encoder_hidden = self.encoder.forward(source)
        decoder_outputs, decoder_hidden = self.decoder.forward(encoder_output,
                                                               (torch.sum(encoder_hidden[0],axis=0,keepdim=True),
                                                                torch.sum(encoder_hidden[1],axis=0,keepdim=True)),
                                                                target.to(device),tf_prob)
        return decoder_outputs, decoder_hidden

if(args.model == 'lstm'):
    seq2seq_model = lstm_model().to(device)
if(args.model == 'lstm_attn'):
    seq2seq_model = lstm_attn_model().to(device)
print('Model Formulated')

###### Training

##### Training Step
def train_epoch(dataloader, model, optimizer, criterion):

    loss_total = 0.0

    for item in tqdm.tqdm(iter(dataloader),colour='blue'):

        source, target = item['data'], item['label']
        model.train()
        optimizer.zero_grad()

        source = source.to(device)
        target = target.to(device)

        with torch.set_grad_enabled(True):
            decoder_outputs,_ = model.forward(source,target,args.tf_prob)
            loss = criterion(decoder_outputs.view(-1,decoder_outputs.size(-1)),
                            target.view(-1))
            
            loss.backward() 
            optimizer.step()

        loss_total += loss.item()*(item['data'].size(0))

    loss_total = loss_total/train_total
    return loss_total

##### Validation Epoch
def val_epoch(dataloader, model, criterion):

    loss_total = 0.0

    for item in tqdm.tqdm(iter(dataloader),colour='green'):

        source, target = item['data'], item['label']
        model.eval()

        source = source.to(device)
        target = target.to(device)
        model = model.to(device)

        with torch.set_grad_enabled(False):
            decoder_outputs,_ = model.forward(source,target,0)
            loss = criterion(decoder_outputs.view(-1,decoder_outputs.size(-1)),
                            target.view(-1))

            loss_total += loss.item()*item['data'].size(0)

    loss_total = loss_total/val_total
    return loss_total

###### Training Validation Loop
def train_val(train_loader, val_loader, model, optimizer, criterion, num_epochs):

    model_path = './Models/'+args.model_name+'.pth'
    loss_best = 1e+6

    for epoch in tqdm.tqdm(range(num_epochs),colour='blue'):

        time_start = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        #### Training
        train_loss_epoch = train_epoch(train_loader,model,optimizer,criterion)
        train_loss.append(train_loss_epoch)

        #### Validation
        val_loss_epoch = val_epoch(val_loader,model,criterion)
        val_loss.append(val_loss_epoch)

        if(val_loss_epoch < loss_best):
            loss_best = val_loss_epoch
            torch.save(model.state_dict(),model_path)

        #### Outputs
        print('Total time:'+str(time.time() - time_start))
        print('Loss: '+str(train_loss_epoch))
        print('Validation Loss: '+str(val_loss_epoch))

    return train_loss, val_loss

###### Training and Validation
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(seq2seq_model.parameters(),lr=1e-4)
train_loss, val_loss = train_val(TrainLoader,ValLoader,seq2seq_model,optimizer,criterion,100)

np.savez_compressed('./Loss/'+args.model_name+'_trainloss.npz',np.array(train_loss))
np.savez_compressed('./Loss/'+args.model_name+'_valloss.npz',np.array(val_loss))
