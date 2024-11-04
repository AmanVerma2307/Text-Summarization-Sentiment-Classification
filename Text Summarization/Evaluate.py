####### Importing Libraries
import os
import torch
import evaluate
import numpy as np
import pandas as pd
from Preprocess import preprocess
from VocabularyGeneration import vocab_gen
from BeamDecode import beam_search

####### Evaluate
def infer_preds(preds,true,idx2word,beam_width,source):

    preds_list = []
    true_list = []
    bleu_list = [] # List to store Bleu scores
    chrf_list = [] # List to store cHRF scores
    true_list = [] # List to store true values
    source_list = [] # List to store source values

    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")

    preds = (beam_search(preds,beam_width)).detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    source = source.detach().cpu().numpy()

    batch_size = int(preds.shape[0]) # Batch size

    for i in range(batch_size):

        preds_curr = []
        true_curr = []
        source_curr = []

        for idx in preds[i]:
            val_curr = idx2word[idx.item()]
            if(val_curr == "<eos>"):
                preds_curr.append(val_curr)
                break
            else:
                preds_curr.append(val_curr)
        preds_list.append(" ".join(preds_curr))

        for idx in true[i]:
            val_curr = idx2word[idx.item()]
            if(val_curr == "<eos>"):
                true_curr.append(val_curr)
                break
            else:
                true_curr.append(val_curr)
        true_list.append(" ".join(true_curr))

        for idx in source[i]:
            val_curr = idx2word[idx.item()]
            if(val_curr == "<eos>"):
                source_curr.append(val_curr)
                break
            else:
                source_curr.append(val_curr)
        source_list.append(" ".join(source_curr))

        bleu_score = bleu.compute(predictions=[" ".join(preds_curr)], references=[" ".join(true_curr)])['bleu']
        chrf_score = chrf.compute(predictions=[" ".join(preds_curr)], references=[" ".join(true_curr)])['score']

        bleu_list.append(bleu_score)
        chrf_list.append(chrf_score)

    return bleu_list, chrf_list, preds_list, true_list, source_list

def evaluate_preds(preds,true,idx2word,beam_decode,beam_width):

    preds_list = []
    true_list = []
    
    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")

    if(beam_decode == False):
        preds = torch.argmax(preds,dim=-1).detach().cpu().numpy()
        true = true.detach().cpu().numpy()

    else:
        preds = (beam_search(preds,beam_width)).detach().cpu().numpy()
        true = true.detach().cpu().numpy()

    #print(preds.shape)
    #print(true.shape)

    batch_size = int(preds.shape[0]) # Batch size

    for i in range(batch_size):

        preds_curr = []
        true_curr = []

        for idx in preds[i]:
            val_curr = idx2word[idx]
            if(val_curr == "<eos>"):
                #preds_curr.append(val_curr)
                break
            else:
                preds_curr.append(val_curr)
        preds_list.append(" ".join(preds_curr))

        for idx in true[i]:
            val_curr = idx2word[idx]
            if(val_curr == "<eos>"):
                #true_curr.append(val_curr)
                break
            else:
                true_curr.append(val_curr)
        true_list.append(" ".join(true_curr))

    bleu_score = bleu.compute(predictions=preds_list, references=true_list)['bleu']*batch_size
    chrf_score = chrf.compute(predictions=preds_list, references=true_list)['score']*batch_size

    return bleu_score, chrf_score, preds_list
