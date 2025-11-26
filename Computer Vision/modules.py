import torch 
import torch.nn as nn 
from torch.nn import functional as F



# ------ BULDING ViT EMBEDDINGS

class PatchEmbedding(nn.Module):
    """    
    First Linear Projection of the Image Patches
    """
    def __init__(self,in_channel,embd_dim,patch_size ):
        super().__init__()
        # A efectos practicos realizar un Convolucionado sobre la imagen total 
        # con kernel_size = patch_size y stride = patch_size
        # es equivalente a:
        # dividir la imagen en los patch correspondientes + projectar linealmente cada uno --> flatten(2) los  concatena
        self.projection = nn.Conv2d(in_channel, embd_dim ,patch_size, stride = patch_size)        
    
    def forward(self, x):
        x = self.projection(x) # (B,C,h,w) : H/patch_size => h; W/patch_size => w;
        x = x.flatten(2) # (B,C,(h·w)) : (h·w) => T => [(H·W)/(patch_size**2)] => N: seq_len (cantidad total de patches)
        x = x.transpose(1,2) # (B,T,C) : ready for the Transformer
        
        return x # (B,T,C)


# ------ BUILDING TRANSFORMER ARQUITECTURE


class FNN(nn.Module):
   def __init__(self, embd_dim,dropout):
       super().__init__()
       self.net = nn.Sequential(
           nn.Linear(embd_dim, 4*embd_dim),
           nn.GELU(),
           nn.Dropout(dropout),
           nn.Linear(4*embd_dim,embd_dim),
           nn.Dropout(dropout)
           )
    
   def forward(self,x):
       x = self.net(x)
       return x # (B,T,C)

class Head(nn.Module):
    def __init__(self,embd_dim,head_size,dropout):
        super().__init__()
        self.query = nn.Linear(embd_dim, head_size, bias=False)
        self.key = nn.Linear(embd_dim, head_size, bias=False)
        self.value = nn.Linear(embd_dim, head_size, bias=False)

        self.drop = nn.Dropout(dropout)
            
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
        s_QK = self.drop(s_QK)
        
        output = s_QK @ v  # (B, T, T) @ (B,T,head_size) = (B,T,head_size)
        output = self.drop(output)

        return output # (B,T,head_size)
        


class MultiheadAttention(nn.Module):

    def __init__(self,embd_dim,n_heads,dropout):

        super().__init__()
        # echa head => (B,T,head_size)
        self.heads = nn.ModuleList([Head(embd_dim,embd_dim//n_heads,dropout) for _ in range(n_heads)]) 
        # estamos formando una lsita con n_heads heads:
        # [(B,T,head_size),(B,T,head_size),...,(B,T,head_size)]
        self.projection = nn.Linear(embd_dim,embd_dim)

        self.drop = nn.Dropout(dropout)

    def forward(self,x):
        # torch.cat agrupara todos los elementos de la lsita de Heads (nn.Modulelist)
        # al rededor de la dimensión que le indiquemos, en este caso, la última dimendion head_size
        # Basicamente estamos concatenando una cabeza junto a la otra y formando matrices conjuntas
        # nuestra tercera/última dimension será ahora head_size*n_heads => por construcción/conveniencia => embd_dim:= C
        out = torch.cat([head(x) for head in self.heads], dim=-1) # (B, T, C)
        out = self.projection(out)
        out = self.drop(out)

        return out # (B,T,C)
     
    

class Block(nn.Module):
    def __init__(self,embd_dim,n_heads,dropout):
        super().__init__()
        self.mh_attention = MultiheadAttention(embd_dim,n_heads,dropout)
        self.fnn = FNN(embd_dim, dropout) 
        self.ln1 = nn.LayerNorm(embd_dim) # especie de BatchNorm pero sobre el block_size, es decir sobre la
        self.ln2 = nn.LayerNorm(embd_dim) # es decri, sobre la dimensión de los tokens (la imagen en este caso)
    
    def forward(self,x):

        x = x + self.mh_attention(self.ln1(x)) # añadimos shortcut (previene vanish gradient en deep NN) 
        x = x + self.fnn(self.ln2(x))  # y hacemos Normalización Previa a entrar

        return x # (B,T,C)
    


# ------ BUILDING FINAL ViT

class Vit(nn.Module):
    def __init__(self,in_channel,embd_dim,patch_size,seq_len,n_heads, n_layers,n_classes,dropout):
        super().__init__()
        self.patch_embd = PatchEmbedding(in_channel,embd_dim,patch_size) # (B,T,C) 
        self.poss_embd = nn.Parameter(torch.zeros(1, seq_len + 1, embd_dim)) # (1,T+1,C); +1 por el CLS token  
        self.blocks = nn.Sequential(*[Block(embd_dim,n_heads,dropout) for _ in range(n_layers)]) # --> la Red recorre: (Block())*n_layer veces
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embd_dim)) # (1,1,C), special token (El Clasificador) 
        self.mlp = nn.Linear(embd_dim,n_classes) # (B,1,Labels)

        self.drop = nn.Dropout(dropout)

    def forward(self,x):

        B,T,H,W = x.shape # input => Imagen (batch,channel,height,width)

        x = self.patch_embd(x)
        # Add CLS token
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) # incoorporamos CLS al tensor
        x = x + self.poss_embd # añadimos positional embeding ((B,T,C) + B*(1,T,C) # torch replicara el pos_embd B times. C = embd_dim
        x = self.drop(x)
        x = self.blocks(x) # cadena de atención
        # IMP: lo que llega al MLP clasificador final no es la sequencia entera
        # es solamente el token CLS que añadimos artificialmente y que 'insorpora'
        # toda la información (junto con las conexiones de atencion) relevantes 
        # para que la maquina pueda clasificar
        x = x[:,0,:] # (B,T+1,C) --> (B,1,C), cogemos solo el token representativo de la imagen (CLS)
        logits = self.mlp(x) # (B,1,Labels)

        return logits



"""

#--MINI PRUEBA DE FUNCIONAMIENTO:

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------- CONFIGURACIÓN DEL MINI TEST -------------

BATCH_SIZE = 8
IMG_CHANNELS = 3
IMG_SIZE = 64
PATCH_SIZE = 16
EMBD_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
NUM_CLASSES = 10

# número de patches de una imagen cuadrada
SEQ_LEN = (IMG_SIZE // PATCH_SIZE) ** 2

# ----------- DATOS FALSOS (para testear) -------------

# imágenes aleatorias
X = torch.randn(64, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)   # 64 imágenes
# etiquetas aleatorias
y = torch.randint(0, NUM_CLASSES, (64,))

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------- DEFINIMOS EL MODELO -------------

model = Vit(
    in_channel=IMG_CHANNELS,
    embd_dim=EMBD_DIM,
    patch_size=PATCH_SIZE,
    seq_len=SEQ_LEN,
    n_heads=NUM_HEADS,
    n_layers=NUM_LAYERS,
    n_classes=NUM_CLASSES
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ----------- LOOP DE ENTRENAMIENTO -------------

print("Verificando shapes y pasos...")

for epoch in range(10):
    for i, (images, labels) in enumerate(loader):

        # Forward
        logits = model(images)             # (B, num_classes)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print(f"Batch {i} | Loss = {loss.item():.4f} | Logits shape = {logits.shape}")

print("✔ Test completado. El modelo forward/backward funciona correctamente.")

"""