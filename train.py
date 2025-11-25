# from modules import Vit
import torch
from torch.utils.data import Dataset , DataLoader
from torchvision.datasets import ImageFolder #para optener los labels en el DataSet de forma sencilla
from PIL import Image

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




# Uso
dataset = Vehicles5Dataset("./VehiclesDataset/dataset")


    


        






