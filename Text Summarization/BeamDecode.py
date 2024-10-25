####### Importing Libraries
import os
import gc
import time
import tqdm
import torch
import argparse

def beam_search(prediction, k=10):

    batch_size, seq_length, vocab_size = prediction.shape # prediction: [N,T,O]
    log_prob, indices = prediction[:, 0, :].topk(k, sorted=True) # indices: [N,1,1]
    indices = indices.unsqueeze(-1) # indices: [N,1]

    for n1 in range(1, seq_length): # Iteration over sequence length

        log_prob_temp = log_prob.unsqueeze(-1) + prediction[:, n1, :].unsqueeze(1).repeat(1, k, 1) 
        log_prob, index_temp = log_prob_temp.view(batch_size, -1).topk(k, sorted=True)
        idx_begin = index_temp // vocab_size  # retrieve index of start sequence
        idx_concat = index_temp % vocab_size  # retrieve index of new token
        new_indices = torch.zeros((batch_size, k, n1+1), dtype=torch.int64)

        for n2 in range(batch_size):
            new_indices[n2, :, :-1] = indices[n2][idx_begin[n2]]
            new_indices[n2, :, -1] = idx_concat[n2]
        indices = new_indices

    return indices, log_prob

#a = torch.randn(2,10,200).float()
#indices, log_prob = beam_search(a,5)
#print(indices.get_device())
#print(indices.shape,log_prob.shape)
#print(indices)
#print(indices[:,0,:])