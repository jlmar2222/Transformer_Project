import torch 
import torch.nn as nn 
from torch.nn import functional as F




# Hypermarameters
batch_size = 4
block_size = 8 # Context Window
max_epoch = 5000
eval_interval = 500
lerning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
heads_num = 4
head_size = n_embd // heads_num
dropout = 0.2
n_layers = 6 
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


# Convert manuscrito into torch.tensor and split it on train vs. val
data = torch.tensor(encode(text), dtype=torch.long)
train_data = data[:int(0.9*len(data))]
val_data = data[int(0.9*len(data)):]



# FUNCTIONS

def get_batch(split):

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    dx = torch.stack([data[i:i+block_size] for i in ix])
    dy = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return dx, dy


# recover mean loss across eval_iters  
@torch.no_grad()
def estimate_loss():
   out = {}

   model.eval()
   for split in ['train', 'val']:      
      losses = torch.zeros(eval_iters) # Write tensor before filling it 
      for iter in range(eval_iters):
         X,Y = get_batch(split)
         logits, loss = model(X,Y)
         losses[iter] = loss.item()

      out[split] = losses.mean()
    
   model.train()
   return out 
         




# MODEL FEATURES

class Head(nn.Module):
  
  def __init__(self):
    super().__init__()

    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size,bias=False)
    #ahora añadimos un buffer que se llama que es como una variable (matriz triangular)
    #un buffer es algo de la arquitectura pero que no es un Parametro entrenable
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.drop = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape

    q = self.query(x) #(B,T,head_size)
    k = self.key(x) #(B,T,head_size)
    v = self.value(x) #(B,T,head_size)

    QK = q@k.transpose(-2,-1) * C**(-0.5)#(B,T,head_size) @ (B,head_size,T)
    QK = QK.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    QK = F.softmax(QK, dim=-1)
    QK = self.drop(QK)

    output = QK @ v  # (B, T, T) @ (B,T,head_size)

    return output
  

class MultiHeadAttention(nn.Module):

    def __init__(self):
       super().__init__()
       self.heads = nn.ModuleList([Head() for _ in range(heads_num)])
       self.proj = nn.Linear(n_embd, n_embd) # Añadimo sprojección para mejorar el entreno
       self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
       out = torch.cat([h(x) for h in self.heads], dim=-1)   
       out = self.proj(out)
       out = self.drop(out)
       return out 
  

class FeedForward(nn.Module):
   def __init__(self):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(n_embd,4*n_embd),
         nn.ReLU(),
         nn.Linear(4*n_embd, n_embd),
         nn.Dropout(dropout)
      )

   def forward(self,x):
      return self.net(x)


class Block(nn.Module):
   def __init__(self):
      super().__init__()
      self.sa_mhead = MultiHeadAttention()
      self.ffn = FeedForward()
      self.lnorm1 = nn.LayerNorm(n_embd)
      self.lnorm2 = nn.LayerNorm(n_embd)
    
   def forward(self, x):     
      # Residual block / short-cut 
      x = x + self.sa_mhead(self.lnorm1(x))  #(Comunicación); normalizamos x antes de entrar en bloque
      # Residual block / short-cut
      x = x + self.ffn(self.lnorm2(x)) # (Computación); normalizamos x antes de entrar en bloque
     
      return x
    



class BigramModel(nn.Module):
  def __init__(self):
    super().__init__()
    # la matriz de pesos que el modelo simple aprenderá
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # ahora esto no nos da directamente los logits
    self.pos_embedding_table = nn.Embedding(block_size,n_embd)
    self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])  # *[Block() for _ in range(n_layers)] --> nn.Sequential (Block(),Block(),...,Block())
    self.lnorm = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd,vocab_size)    

  def forward(self, idx, targets=None):
      
      B,T = idx.shape
      
      tok_emb = self.token_embedding_table(idx) # (B,T,numb_embd)
      pos_emb = self.pos_embedding_table(torch.arange(T,device=device)) # (T, numb_embd) Las posiciones son iguales para todos los Batches
                                        # torch.arrenge(T), generates a tensor with de positions [0,1,...,T]
      
      x = tok_emb + pos_emb
      x = self.blocks(x)
      x = self.lnorm(x)
      logits = self.lm_head(x) # (B,T,vocab_size)

      if targets is None:
        loss = None
      else:
        B, T, C = logits.shape
        logits = logits.view(B*T, C) 
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
      
      return logits, loss

  def generate(self, idx, max_new_tokens):
         
        for _ in range(max_new_tokens):
            # adjust lenght of the idx to match max_size of block_size
            idx_cond = idx[:,-block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            #reducimos y solo cogemos ultima isntancia temporal (modelo Bigramo)
            logits = logits[:,-1,:] # (B,C)
            # convertimos los valores a probabilidades con una softmax
            probs = F.softmax(logits, dim=1) # (B,C)
            # tiramo sun draw de una multinomial con esas probabilidades (con una multinomial)
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # la concatenamos 
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx



#---------------------------

model = BigramModel()
m = model.to(device)

# set the PyTorch optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=lerning_rate)

# set the training process

for epoch in range(max_epoch):    

   # print some evaluation stuff
    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


   # Actual optimization:

        # sample a batch of data
    xb, yb = get_batch('train')

        # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step() 


# finall generate results:

user_input = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(user_input, max_new_tokens=1000)[0].tolist()))