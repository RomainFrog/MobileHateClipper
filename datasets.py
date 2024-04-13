import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset

class HatefulMemesDataset(Dataset):
    def __init__(self, data_dir, img_dir, split='train', img_size=256):
        self.data_path = data_dir
        self.img_dir = img_dir
        self.split = split
        self.img_size = img_size
        self.data = [json.loads(line) for line in open(os.path.join(data_dir, f'{split}.jsonl'), 'r')]


    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.data[idx]['img']
        img_path = os.path.join(self.img_dir, img)
        img = Image.open(img_path).convert('RGB').resize((self.img_size, self.img_size))
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        return img, text, label
    
