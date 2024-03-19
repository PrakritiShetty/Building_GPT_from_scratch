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


"""LANGUAGE MODEL - BIGRAM"""
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    # creating a token embedding table such that each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)  
     

  def forward(self, idx, targets=None):
    # idx and targets are both (0,T) tensor of ints
    logits = self.token_embedding_table(idx) #(B,T,C) 
    # every one of our input will lookup the embedding table and pluck out a row corresp to that idx. Then pytorch will assemble all of that into a batch*channel*time tensor
    # this is logits or score for next char in sequence - hence we have successfully predicted next char in sequence
    

    # directly loss = F.cross_entropy(logits, targets) won't work, because pytorch cross entropy documentation says that if you have a multi dim input, should be B*C*T and not B*T*C

    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape # unpacks numbers
      logits = logits.view(B*T, C) #stretches out B and T dims as one, and preserves channel dimensions
      #similarly, targets is B,T and we want it to be B*T - 1D
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss


m = BigramLanguageModel()

context = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(context,max_new_tokens=400)[0].tolist()))