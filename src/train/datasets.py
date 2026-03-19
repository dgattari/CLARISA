
# src/train/datasets.py  

from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

from .legacy_train_base import get_image_cached # el base me lo voy a cargar
# el get_image_cached no se desde donde lo podemos importar pero no me gusta en base. Igual
# estaría bien meterlo aquí.

from src.preprocessing.crops import crop_center

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

class MultiInputROIDataset(Dataset):
    """
    Dataset multi-input para entrenamiento del clasificador.

    Modos soportados:
      - '256'  : crop centrado 256x256
      - '384'  : crop centrado 512x512 -> resize a 384
      - 'stack': concatena ambas vistas (256 + 512) en 6 canales
    """
    def __init__(self, samples: List[Dict[str,Any]], input_mode: str, augment: bool, resize_to: int = 384):
        self.samples = samples
        self.mode = input_mode
        self.augment = augment
        self.resize_to = resize_to

        if augment: # aqui se pueden estudiar los augmentations que dijimos 
            self.tf = A.ReplayCompose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    scale=(0.9, 1.1),
                    rotate=(-10, 10),
                    p=0.3,
                ),
                A.Resize(resize_to, resize_to, interpolation=1),
            ])
        else:
            self.tf = A.ReplayCompose([
                A.Resize(resize_to, resize_to, interpolation=1),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = get_image_cached(Path(s['image_path']))
        
        (x1,y1,x2,y2) = s['bbox']
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if self.mode == '256':
            roi = crop_center(img, cx, cy, 256)
            out = self.tf(image=roi)
            im = out['image']
            
            t = torch.from_numpy(im).permute(2,0,1).float() / 255.0
            t = (t - IMAGENET_MEAN) / IMAGENET_STD
            
            label = int(s['label'])
            return t, label

        elif self.mode == '384':
            # crop grande 512 -> resize 384
            roi = crop_center(img, cx, cy, 512) # cambiar
            out = self.tf(image=roi)
            
            im = out['image']
            
            t = torch.from_numpy(im).permute(2,0,1).float() / 255.0
            t = (t - IMAGENET_MEAN) / IMAGENET_STD
            
            label = int(s['label'])
            return t, label

        elif self.mode == 'stack':
            roi256 = crop_center(img, cx, cy, 256)
            roi384 = crop_center(img, cx, cy, 512) # cambiar
            
            out = self.tf(image=roi256)
            im256 = out['image']
            im384 = A.ReplayCompose.replay(out['replay'], image=roi384)['image']
            
            t256 = torch.from_numpy(im256).permute(2,0,1).float() / 255.0
            t384 = torch.from_numpy(im384).permute(2,0,1).float() / 255.0
            
            t256 = (t256 - IMAGENET_MEAN) / IMAGENET_STD
            t384 = (t384 - IMAGENET_MEAN) / IMAGENET_STD
            
            t = torch.cat([t256, t384], dim=0)
            label = int(s['label'])
            return t, label

        else:
            raise ValueError("input_mode debe ser '256', '384' o 'stack'")

# me gustaría que esto tuviera su script individual
def subset_samples(samples: List[Dict[str, Any]], indices) -> List[Dict[str, Any]]:
    """
    NEW refactorization from Dani's code: Devuelve una sublista de samples usando un array/lista de índices.
    """
    return [samples[i] for i in indices]

# make_loaders(...)
# - en datasets.py
# - porque depende del dataset y transforms

def make_loaders(
    samples: List[Dict[str, Any]],
    tr_idx,
    va_idx,
    input_mode: str,
    batch_size: int,
    num_workers: int,
    resize_to: int,
):
    """
    New Updated code but Dani's core function kept. Construye DataLoaders de train y validation a partir de índices.

    Esta función se usa desde trainer.py para desacoplar:
      - protocolo de split
      - construcción del dataset
      - entrenamiento
    """
    tr_samples = subset_samples(samples, tr_idx)
    va_samples = subset_samples(samples, va_idx)

    train_loader = DataLoader(
        MultiInputROIDataset(
            tr_samples,
            input_mode=input_mode,
            augment=True,
            resize_to=resize_to,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        MultiInputROIDataset(
            va_samples,
            input_mode=input_mode,
            augment=False,
            resize_to=resize_to,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader

def make_eval_loader(
    samples: List[Dict[str, Any]],
    indices,
    input_mode: str,
    batch_size: int,
    num_workers: int,
    resize_to: int,
):
    """
    NEW function: Loader de evaluación/inferencia sin augmentación.
    Útil para test final.
    """
    eval_samples = subset_samples(samples, indices)

    loader = DataLoader(
        MultiInputROIDataset(
            eval_samples,
            input_mode=input_mode,
            augment=False,
            resize_to=resize_to,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader

