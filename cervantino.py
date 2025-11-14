import torch 
import torch.nn as nn 
from torch.nn import functional as F




# Hypermarameters
batch_size = 32
block_size = 8 # COntext Window
max_iter = 3000
eval_interval = 300
lerning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
#---------------------


torch.manual_seed(1337)

# get manuscrito
with open('don_quijote_normalizado.txt', 'r', encoding='utf-8') as f:
    text = f.read()


vocab = sorted(list(set(text)))
vocab_size = len(vocab)


# Creating de Encoder/Decoder
stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }
encode = lambda frase: [stoi[c] for c in frase]
decode = lambda enumeracion: ''.join([itos[i] for i in enumeracion])


