###### Importing Libraries
import os
import gc
import time
import tqdm
import torch
import argparse
import numpy as np
import pandas as pd
from Preprocess import preprocess
from DataLoader import SentimentSet
from SentimentRNN import SentimentRNN
from sklearn.metrics import classification_report

###### Input arguments
parser = argparse.ArgumentParser()

parser.add_argument("--csv_file",
                    type=str,
                    help="Path to the input csv file")
parser.add_argument("--model_path",
                    type=str,
                    help="Path to the model")
parser.add_argument("--output_file",
                    type=str,
                    help="Path to the output csv file")

args = parser.parse_args()

###### Preprocessing
X, y = preprocess(args.csv_file)

###### Dataset building

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
y_preds = [] # List to store output

##### Dataset and DataLoader
Dataset = SentimentSet(X,y,vocab,max_seq_len,pad_token,unk_token)
del(X)
gc.collect()
loader = torch.utils.data.DataLoader(Dataset,batch_size=1,
                                           shuffle=True,
                                           drop_last=False)

####### Model Evaluation

##### Model
device = torch.device("cuda:0")
model = SentimentRNN(vocab_size,embedding_dim,100,1,3)
model.load_state_dict(torch.load(args.model_path))
model = model.to(device)
model.eval()

##### Evaluation loop
for item in tqdm.tqdm(iter(loader),colour='blue'):

    with torch.set_grad_enabled(True):
        outputs = model(item['data'].to(device),(model.init_hidden(int(item['data'].size(0)))).to(device))

    outputs = torch.argmax(outputs,dim=-1).cpu().detach().numpy()

    for pred_val in outputs:
        y_preds.append(pred_val) 

##### Metrics
print(classification_report(y_pred=y_preds,y_true=y))

###### Class2Text
y_preds_txt = []
for item in y_preds:
    if(item == 0):
        y_preds_txt.append('Positive')
    if(item == 1):
        y_preds_txt.append('Neutral')
    if(item == 2):
        y_preds_txt.append('Negative')

##### Saving outputs
tweet_df = pd.read_csv(args.csv_file,index_col=False)
tweet_df  = pd.DataFrame(tweet_df[['tweet_id', 'text', 'airline_sentiment']])
tweet_df['predicted'] = y_preds_txt
tweet_df.to_csv(args.output_file)
