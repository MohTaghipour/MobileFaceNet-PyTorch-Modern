# LFW_loader.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

class LFW(Dataset):
    def __init__(self, left_paths, right_paths, input_size=112):
        self.left_paths = left_paths
        self.right_paths = right_paths
        self.input_size = input_size
        # Same transform as training (but no random flip!)
        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),                                 # → [0,1] + HWC → CHW
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # → [-1, 1]
        ])

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, index):
        # Load left image
        img_l = Image.open(self.left_paths[index]).convert('RGB')
        img_r = Image.open(self.right_paths[index]).convert('RGB')
        # Apply transform
        img_l = self.transform(img_l)
        img_r = self.transform(img_r)
        # Stack as (2, 3, 112, 112)
        batch = torch.stack([img_l, img_r])
        return batch  # shape: (2, 3, 112, 112)

if __name__ == '__main__':
    pass