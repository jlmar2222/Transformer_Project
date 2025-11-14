import torch 
import torch.nn as nn 
from torch.nn import functional as F




# Hypermarameters
batch_size = 32
block_size = 8 # COntext Window
max_epoch = 3000
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

class BigramModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    # la matriz de pesos que el modelo simple aprenderá
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
      logits = self.token_embedding_table(idx)

      if targets is None:
        loss = None
      else:
        B, T, C = logits.shape
        logits = logits.view(B*T, C) 
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
      
      return logits, loss

  def generate(self, idx, max_new_tokens):
        # 
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
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

model = BigramModel(vocab_size)
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
print(decode(m.generate(user_input, max_new_tokens=100)[0].tolist()))