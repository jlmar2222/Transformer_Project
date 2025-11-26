from modules import Vit # nuestro modelo

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from torch.utils.data import Dataset , DataLoader

from torchvision.datasets import ImageFolder #para optener los labels en el DataSet de forma sencilla
from PIL import Image
from torchvision import transforms 

from tqdm import tqdm

import numpy as np

import kagglehub
import shutil
import os
import zipfile
from pathlib import Path


# ------ DOWNLOAD DATA

data_dir = "./"
final_name = "VehiclesDataset"
final_path = os.path.join(data_dir, final_name)

if not os.path.exists(final_path):

    os.makedirs(data_dir, exist_ok=True)
    # Descargar dataset
    path = kagglehub.dataset_download("mrtontrnok/5-vehichles-for-multicategory-classification")
    print("Descargado en:", path)
    # Si ya existía la carpeta final, la borramos
    if os.path.exists(final_path):
        shutil.rmtree(final_path)
    # Mover todo el contenido a nuestra carpeta final
    shutil.move(path, final_path)

    print("Dataset preparado en:", final_path)

    # Cleaning data (resolviendo algun error de la base de datos => solo nos quedamos con archivos .png)
    root = "./VehiclesDataset/dataset"   # tu carpeta raíz
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
                # si no termina en .png -> eliminar
            if not filename.lower().endswith(".png"):
                    file_path = os.path.join(dirpath, filename)
                    print(f"Eliminando: {file_path}")
                    os.remove(file_path)
    


else:
    pass



# ------ BULDING TORCH DATASET

class Vehicles5Dataset(Dataset):
    def __init__(self, data_dir, split = 'train',transform=None):
        #self.data_dir = Path(data_dir)
        self.transform = transform
        

        if split == 'train':
            self.data_dir = Path(data_dir) / 'train'
        else:
            self.data_dir = Path(data_dir) / 'test'


        self.img_paths = []
        self.labels = []
        
        self.classes = ImageFolder(root=self.data_dir).classes # devuelve ['car','bus',...,'bike']        
        self.classes_idx = ImageFolder(root=self.data_dir).class_to_idx # devulve {'car': 0, 'bus': 1,.., 'bike': 4}

        # Recorrer subcarpetas
        for class_name in self.classes:
            folder = self.data_dir / class_name
            for img_file in folder.iterdir():  # recorre cada imagen
                self.img_paths.append(img_file)
                self.labels.append(self.classes_idx[class_name]) # mappeamos de la class_name a su Número (target)
                

        print("Clases detectadas:", self.classes)
        print("Número de imágenes:", len(self.img_paths))        


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        # para tratmeinto de las imagenes
        img = Image.open(img_path).convert("RGB")

        if self.transform: # solo transformaremos en periodo de entrenamiento no en validación
            img = self.transform(img)

        return img, label



# Mix Up es un metodo que de forma random 
# unifica/mezcla con cierto porcentaje o factor lam 
# dos datos ()en este caso imagenes
# Mejora la generalización del modelo, como otros metodos que aplicaremos tambien 

def mixup_data(x, y): # x --> data, y --> target/lables
    lam = np.random.beta(0.2, 0.2)

    batch_size = x.size(0) # x = tensor([batch_size, chanels, height, width]); x.size(0) = batch size
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Cuando se aplique el MixUp tendremos que modificar ligeramente la loss que le asignamos en el entreno
def mixup_criterion(criterion, pred, y_a, y_b, lam): # compute loss criteria under mixed stuff
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)




# --- Hyperparameters:

in_channel = 3
embd_dim = 32
img_size = 256 # debido al resize que aplicamos a todas las imagenes
patch_size = 16
seq_len = (img_size//patch_size)**2
n_heads = 4
n_layers = 4
n_classes = 5
dropout = 0.2

def train():
  
    data_path = "./VehiclesDataset/dataset"

    # Más transformaciones:
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),      # igualamos tamaños
        transforms.RandomHorizontalFlip(),# voltea imágenes aleatoriamente
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),      # gira hasta ±10 grados
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2),  # cambios leves en luz y contraste
        transforms.ToTensor(),               # pasa a tensor (C x H x W)
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # Normalizar valores entre -1 y 1 
    ])


    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),      # igualamos tamaños
        transforms.ToTensor(),               # pasa a tensor (C x H x W)
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # Normalizar valores entre -1 y 1 
    ])

    # Declaring Data Sets 
    train_dataset = Vehicles5Dataset(data_dir=data_path, split='train', transform=train_transform)
    test_dataset = Vehicles5Dataset(data_dir=data_path, split='test', transform=test_transform)

    print(f'Train Data Set = {len(train_dataset)}')
    print(f'Test Data Set = {len(test_dataset)}')

    # Estableciendo torch Data Loader (por conveniencia y eficiencia interna)
    train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Vit(in_channel,embd_dim,patch_size,seq_len,n_heads, n_layers,n_classes,dropout)
    model = model.to(device)
    
    num_epochs = 100
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1
        )

    best_accuracy = 0 # vamos a guardar los parametros del modelo con mejor accuracy


    print('Start Training')

    for epoch in range(num_epochs):

        # Training Mode:

        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for data, target in progress_bar:

            data, target = data.to(device), target.to(device)
            # Condicionamos a MixUP
            if np.random.random() > 0.7: # es decir un 0.3 de probabilidades
                data, target_a, target_b, lam = mixup_data(data,target)
                pred = model(data)
                loss = mixup_criterion(criterion, pred, target_a, target_b, lam)
            else:
                pred = model(data)
                loss = criterion(pred,target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() # .item() para hacerlo un int que python pueda leer
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})            
        
        
        avg_train_epoch_loss = epoch_loss/len(train_dataloader) # esto nos da la loss media por batch (B)


        # Test Model
        model.eval()

        correct = 0
        total = 0
        test_loss = 0

        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)

                pred = model(data)
                loss = criterion(pred, target)

                test_loss += loss.item()

                _,prediction = torch.max(pred.data,1)
                total += target.size(0) # tamaño de muestras por Batch 
                correct += (prediction == target).sum().item()

        accuracy = 100 * correct / total
        avg_test_loss = test_loss/len(test_dataloader)


        print(f'Epoch {epoch+1} Loss: {avg_train_epoch_loss:.4f}, Val Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'accuracy': accuracy,
                    'epoch': epoch,
                    'classes': train_dataset.classes
                }, 'best_model.pth')
            print(f'New best model saved: {accuracy:.2f}%')



train()




  
        






