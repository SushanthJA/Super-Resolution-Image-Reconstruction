import os
import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import io, transforms
import torchvision.transforms.functional as F

# Defining the class for data augmentation techniques and a custom_collate_function to stack HR and LR patches together
class PairedTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, hr, lr):
        for transform in self.transforms:
            hr, lr = transform(hr, lr)
        return hr, lr

class RandomHorizontalFlip:
    def __call__(self, hr, lr):
        if random.random() < 0.5:
            hr = F.hflip(hr)
            lr = F.hflip(lr)
        return hr, lr

class RandomVerticalFlip:
    def __call__(self, hr, lr):
        if random.random() < 0.5:
            hr = F.vflip(hr)
            lr = F.vflip(lr)
        return hr, lr

class RandomRotationSpecific:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, hr, lr):
        angle = random.choice(self.angles)
        hr = F.rotate(hr, angle)
        lr = F.rotate(lr, angle)
        return hr, lr

def custom_collate_fn(batch):
    hr_patches, lr_patches = zip(*batch)
    hr_patches = torch.stack(hr_patches)
    lr_patches = torch.stack(lr_patches)
    
    # print(f"Batch HR Shape: {hr_patches.shape}, Batch LR Shape: {lr_patches.shape}") # Debug
    
    return hr_patches, lr_patches


# Defining the dataloader class
class SRDataset(Dataset):
    def __init__(self, hr_patch_dir, lr_patch_dir, paired_transform=None, tensor_transform=None, paired_transform_prob=0.2):
        self.hr_patch_dir = hr_patch_dir
        self.lr_patch_dir = lr_patch_dir
        self.paired_transform = paired_transform
        self.tensor_transform = tensor_transform
        self.paired_transform_prob = paired_transform_prob
        self.hr_patches = sorted([f for f in os.listdir(hr_patch_dir) if f.endswith('.png')])
        self.lr_patches = sorted([f for f in os.listdir(lr_patch_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.hr_patches)
    
    def __getitem__(self, idx):
        hr_patch = os.path.join(self.hr_patch_dir, self.hr_patches[idx])
        lr_patch = os.path.join(self.lr_patch_dir, self.lr_patches[idx])
        
        hr_patch = Image.open(hr_patch).convert("RGB")
        lr_patch = Image.open(lr_patch).convert("RGB")
        
        if self.paired_transform and random.random() <= self.paired_transform_prob:
            hr_patch, lr_patch = self.paired_transform(hr_patch, lr_patch)
            
        if self.tensor_transform:
            hr_patch = self.tensor_transform(hr_patch)
            lr_patch = self.tensor_transform(lr_patch)
            
        # print(f"HR Patch Shape: {hr_patch.shape}, LR Patch Shape: {lr_patch.shape}")  # Debugging line
        
        return hr_patch, lr_patch