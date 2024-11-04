####### Importing Libraries
import os
import gc
import time
import random
import argparse
import tqdm
import torch
import evaluate
import numpy as np
import pandas as pd
from Preprocess import preprocess
from VocabularyGeneration import vocab_gen
from DataLoader import SummarySet
from model import EncoderRNN, DecoderRNN
from model_attention import AttnDecoderRNN
from Evaluate import infer_preds

####### Input arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model_file',
                    type=str,
                    help="Path to the model")
parser.add_argument('--beam_size',
                    default=5,
                    type=int,
                    help='Beam width parameter')
parser.add_argument('--model_type',
                    type=str,
                    help="Choice of model: lstm_lstm or lstm_lstm_attn")
parser.add_argument('--max_len',
                    default=10,
                    type=int,
                    help="Maximum length")
parser.add_argument('--test_data_file',
                    type=str,
                    help='Path to the input file')
parser.add_argument('--output_file',
                    type=str,
                    help='Path to the output file')
parser.add_argument('--inference_style',
                    type=str,
                    default='single',
                    help='full: All the beam width, single: The selected beam width')


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
X, y = preprocess(args.test_data_file)
#X_val, y_val = preprocess('./val.csv')

Dataset = SummarySet(X,y,word2idx,
                          idx2word,
                          args.max_len,
                          "<sos>","<eos>")
total = len(X)
del(X)
gc.collect()

##### Dataloader
DataLoader = torch.utils.data.DataLoader(Dataset,
                                   batch_size=128,
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

if(args.model_type == 'lstm_lstm'):
    seq2seq_model = lstm_model()
    seq2seq_model.load_state_dict(torch.load(args.model_file))
    seq2seq_model = seq2seq_model.to(device)
if(args.model_type == 'lstm_lstm_attn'):
    seq2seq_model = lstm_attn_model()
    seq2seq_model.load_state_dict(torch.load(args.model_file))
    seq2seq_model = seq2seq_model.to(device)
print('Model Formulated')

###### Model Evaluation

##### Evaluation loop
def eval(dataloader, model, beam_width):

    bleu_score_total = []
    chrf_score_total = []
    predictions = []
    true = []
    source_store = []

    for item in tqdm.tqdm(iter(dataloader),colour='green'):

        source, target = item['data'], item['label']
        model.eval()

        source = source.to(device)
        target = target.to(device)
        model = model.to(device)

        with torch.set_grad_enabled(False):
            decoder_outputs,_ = model.forward(source,target,0)
            #loss = criterion(decoder_outputs.view(-1,decoder_outputs.size(-1)),
            #                target.view(-1))

            #loss_total += loss.item()*item['data'].size(0)
            bleu_score_curr, chrf_score_curr, preds_curr, true_curr, source_curr = infer_preds(decoder_outputs,target,idx2word,beam_width,source)

            bleu_score_total = bleu_score_total + bleu_score_curr
            chrf_score_total = chrf_score_total + chrf_score_curr
            predictions = predictions + preds_curr
            true = true + true_curr
            source_store = source_store + source_curr

    #loss_total = loss_total/val_total
    #bleu_score_total = bleu_score_total/total
    #chrf_score_total = chrf_score_total/total

    return bleu_score_total, chrf_score_total, predictions, true, source_store

if(args.inference_style == 'single'):
    
    bleu_score, chrf_score, predictions, true, source = eval(DataLoader,seq2seq_model,args.beam_size)

    #review_df = pd.read_csv(args.test_data_file,index_col=False)
    #review_df  = pd.DataFrame(review_df[['Text', 'Summary']])
    review_df = {'Reviews':source, 'Summary':true}
    review_df = pd.DataFrame(data=review_df)
    review_df['Predicted'] = predictions
    review_df['BLEU'] = bleu_score
    review_df['CHRF'] = chrf_score
    review_df.to_csv(args.output_file)

if(args.inference_style == 'full'):

    bleu_5, chrf_5, preds_5, true, source  = eval(DataLoader,seq2seq_model,5)
    bleu_10, chrf_10, preds_10, true, source = eval(DataLoader,seq2seq_model,10)
    bleu_15, chrf_15, preds_15, true, source = eval(DataLoader,seq2seq_model,15)
    bleu_20, chrf_20, preds_20, true, source = eval(DataLoader,seq2seq_model,20)

    #review_df = pd.read_csv(args.test_data_file,index_col=False)
    #review_df  = pd.DataFrame(review_df[['Text', 'Summary']])
    review_df = {'Reviews':source, 'Summary':true}
    review_df = pd.DataFrame(data=review_df)
    
    review_df['Preds_5'] = preds_5
    review_df['Preds_10'] = preds_10
    review_df['Preds_15'] = preds_15
    review_df['Preds_20'] = preds_20

    review_df['BLEU_5'] = bleu_5
    review_df['BLEU_10'] = bleu_10
    review_df['BLEU_15'] = bleu_15
    review_df['BLEU_20'] = bleu_20

    review_df['CHRF_5'] = chrf_5
    review_df['CHRF_10'] = chrf_10
    review_df['CHRF_15'] = chrf_15
    review_df['CHRF_20'] = chrf_20

    review_df.to_csv(args.output_file)    