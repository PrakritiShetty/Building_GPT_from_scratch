import torch
import torch.nn as nn
from torch.nn import functional as F


"""HYPERPARAMETERS"""
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
n_embed = 32 
n_head = 4
n_layer = 3


"""READ DATA"""
with open('input.txt','r',encoding='utf-8') as f:
  text = f.read()

# we now have all the data input as a string
# the next step will be to tokenize it
  
"""TOKENIZER"""
# the simplest way to tokenise text is to use character-level tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)
# creating a mapping from char to int
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
# encoder - takes a string, outputs a list of ints
encode = lambda s: [stoi[c] for c in s]
# decoder - takes a list of ints, outputs a string
decode = lambda l:''.join([itos[i] for i in l])