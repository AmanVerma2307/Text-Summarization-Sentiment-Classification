###### Importing Libraries
import os
import gc
import time
import tqdm
import torch
import numpy as np
import pandas as pd
from Preprocess import preprocess
from DataLoader import SentimentSet
from SentimentRNN import SentimentRNN
print('Importing done')

###### Data Processing
X_train, y_train = preprocess('./train.csv') 
X_val, y_val = preprocess('./val.csv')
print('Preprocessing done')

###### DataLoader

##### Vocab generation
vocab = []
with open('./glove.twitter.27B/glove.twitter.27B.100d.txt','rt',encoding="utf8") as fi:
    full_content = fi.read().strip().split('\n')
for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    vocab.append(i_word)

vocab = np.array(vocab)
vocab = np.insert(vocab, 0, '<pad>')
vocab = np.insert(vocab, 1, '<unk>')

##### Defining Essentials
max_seq_len = 15
pad_token = '<pad>'
unk_token = '<unk>'
vocab_size = 1193516
embedding_dim = 100
train_total = 8000
val_total = 1500

##### Dataset
train_Dataset = SentimentSet(X_train,y_train,vocab,max_seq_len,pad_token,unk_token)
val_Dataset = SentimentSet(X_val,y_val,vocab,max_seq_len,pad_token,unk_token)

del(X_train,X_val)
gc.collect()

##### DataLoader
train_loader = torch.utils.data.DataLoader(train_Dataset,
                                           batch_size=64,
                                           shuffle=True,
                                           drop_last=False)
val_loader = torch.utils.data.DataLoader(val_Dataset,
                                           batch_size=64,
                                           shuffle=True,
                                           drop_last=False)

device = torch.device("cuda:0") # Device placement
print('Dataloader set')

####### Model training

###### Model
model = SentimentRNN(vocab_size,embedding_dim,100,1,3)
model = model.to(device)
print('Model formulated')

###### Defining essentials
train_loss = []
val_loss = []

###### Training Loop
def train_val(model, criterion, optimizer, num_epochs):

    model_path = './part_1.2_rnn_pytorch.pth'
    loss_best = 1e+6

    ##### Training loop
    for epoch in tqdm.tqdm(range(num_epochs),colour='green'):
        
        time_start = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        #### Training
        loss = 0.0
        acc = 0.0

        for item in tqdm.tqdm(iter(train_loader),colour='blue'):

            model.train()
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                outputs = model(item['data'].to(device),(model.init_hidden(int(item['data'].size(0)))).to(device))
                loss_curr = criterion(outputs,(item['label'].type(torch.LongTensor)).to(device))

                loss_curr.backward()
                optimizer.step()

            loss += loss_curr.item()*item['data'].size(0)
            acc += torch.sum(torch.argmax(outputs,dim=-1) == item['label'].to(device)).cpu().detach()

        loss = loss/train_total
        train_loss.append(loss)

        #### Validation
        loss_val = 0.0
        acc_val = 0.0

        for item in tqdm.tqdm(iter(val_loader),colour='blue'):

            model.eval()
            with torch.set_grad_enabled(False):
                outputs = model(item['data'].to(device),(model.init_hidden(int(item['data'].size(0))).to(device)))
                loss_curr = criterion(outputs,(item['label'].type(torch.LongTensor)).to(device))

            loss_val += loss_curr.item()*item['data'].size(0)
            acc_val += torch.sum(torch.argmax(outputs,dim=-1) == item['label'].to(device)).cpu().detach()

        loss_val = loss_val/val_total
        val_loss.append(loss_val)

        if(loss_val < loss_best):
            loss_best = loss_val
            torch.save(model.state_dict(),model_path)

        ##### Outputs
        print('Total time:'+str(time.time() - time_start))
        print('Loss: '+str(loss))
        print('Accuracy: '+str((acc/train_total)*100))
        print('Validation Loss: '+str(loss_val))
        print('Validation Accuracy: '+str((acc_val/val_total)*100))

###### Training and validation
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
model = train_val(model,criterion,optimizer,num_epochs=200)

np.savez_compressed('./train_loss_torch.npz',np.array(train_loss))
np.savez_compressed('./val_loss_torch.npz',np.array(val_loss))
