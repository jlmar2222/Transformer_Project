import torch 
import torch.nn as nn 
from torch.nn import functional as F



# ------ BULDING ViT EMBEDDINGS

class PatchEmbedding(nn.Module):
    """    
    First Linear Projection of the Image Patches
    """
    def __init__(self,in_channel,emb_dim,patch_size ):
        super().__init__()
        # A efectos practicos realizar un Convolucionado sobre la imagen total 
        # con kernel_size = patch_size y stride = patch_size
        # es equivalente a:
        # dividir la imagen en los patch correspondientes + projectar linealmente cada uno --> flatten(2) los  concatena
        self.projection = nn.Conv2d(in_channel, emb_dim ,patch_size, stride = patch_size)        
    
    def forward(self, x):
        x = self.projection(x) # (B,C,h,w) : H/patch_size => h; W/patch_size => w;
        x = x.flatten(2) # (B,C,(h·w)) : (h·w) => T => [(H·W)/(patch_size**2)] => N: seq_len (cantidad total de patches)
        x = x.transpose(1,2) # (B,T,C) : ready for the Transformer
        return x 



class PositionalEmbedding(nn.Module):
    """
    Adds learnable positional embeddings to the inputs.
    """
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim)) # (1,T,C) 
        # recuerda seq_len := T  => es el numero total de patches formados  
            
    def forward(self, x):
        return x + self.embedding # Patch_Embedding + Positional = (B,T,C) + B*(1,T,C) # torch replicara el pos_embd B times.
        # (B,T,C) donde C = embd_dim


    # mas uno (+ 1) que creo que se refiere al CLS token que meten artificialmente.


# ------ BUILDING TRANSFORMER ARQUITECTURE


class FNN(nn.Module):
   def __init__(self, embd_dim):
       super().__init__()
       self.net = nn.Sequential(
           nn.Linear(embd_dim, 4*embd_dim),
           nn.GELU(),
           nn.Linear(4*embd_dim,embd_dim)
           )
    
   def forward(self,x):
       x = self.net(x) # (B,T,C)

class Head(nn.Module):
    def __init__(self,embd_dim,head_size):
        super().__init__()
        self.query = nn.Linear(embd_dim, head_size, bias=False)
        self.key = nn.Linear(embd_dim, head_size, bias=False)
        self.value = nn.Linear(embd_dim, head_size, bias=False)
    
    def forward(self,x):

        B,T,C = x.shape

        q = self.query(x) #(B,T, head_size)
        k = self.key(x) #(B,T, head_size)
        v = self.value(x) #(B,T, head_size)

        # Scale Dot Product Attetion
        QK = q @ k.transpose(-2,-1) # (B,T, head_size) @ (B,head_size,T) = (B,T,T)
        scale = k.size(-1) # scale = head_size / keys_dimention
        scaled_QK = QK * scale**(-0.5) # (B,T,T) --> # para prevenir vanishing gradient
        s_QK = F.softmax(scaled_QK, dim=-1)
        
        output = s_QK @ v  # (B, T, T) @ (B,T,head_size) = (B,T,head_size)

        return output
        


class Multihead(nn.Module):
    pass

class Block(nn.Module):
    pass



       