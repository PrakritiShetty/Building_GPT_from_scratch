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

"""Train-Test Split"""
data = torch.tensor(encode(text),dtype=torch.long)
# splitting our dataset into train and validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

"""data loading"""
def get_batch(split):
  data = train_data if split=='train' else val_data
  # generating random chunks
  # randomly offset numbers between len(data) and block_size. Number of such numbers is batch_size
  ix = torch.randint(len(data) - block_size, (batch_size,))
  # x is the first block size chars starting at i
  x = torch.stack([data[i:i+block_size] for i in ix]) 
  # y is x, but offset by 1
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x,y = x, y
  return x,y
