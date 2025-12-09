import numpy as np
import os
import torch
from config import CASIA_DATA_DIR
from PIL import Image
import torchvision.transforms as T

class CASIA_Face(object):
    def __init__(self, root):
        self.root = root
        img_txt_path = os.path.join(root, 'casia-webface.txt')
        if not os.path.exists(img_txt_path):
            raise FileNotFoundError(f"Could not find {img_txt_path}")

        image_list = []
        label_list = []
        # -------------- First pass: collect unique labels -------------- 
        unique_labels = set()
        with open(img_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue         # skip corrupted lines
                old_label = int(parts[0])
                unique_labels.add(old_label)

        # Map old labels → continuous indices 0 → num_classes-1
        label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
        self.class_nums = len(label_map)            # number of classes in the text file

        # -------------- Second pass: build remapped label list --------------
        with open(img_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue         # skip corrupted lines
                old_label = int(parts[0])           # column 1 → class id → e.g. 1380
                label = label_map[old_label]        # change with continuous label
                rel_path = parts[1]                 # column 2 → image relative path → e.g. casia-webface/000000/00000001.jpg
                age = float(parts[2])               # column 3 → Age → e.g. 28.5
                gender = int(parts[3])              # column 4 → Gender → e.g. 1=male, 0=female
                full_path = os.path.join(root, rel_path)
                if os.path.exists(full_path):       # avoids broken paths
                    image_list.append(full_path)
                    label_list.append(label)

        self.image_list = image_list
        self.label_list = label_list

        print(f"Loaded {len(image_list)} images from {len(np.unique(label_list))} identities.")

        # -------------- Image Transform --------------
        self.transform = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label 

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    dataset = CASIA_Face(root=CASIA_DATA_DIR)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False)
    print(f"Data set size: {len(dataset)}")