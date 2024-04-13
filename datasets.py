import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
import mobileclip

class HatefulMemesDataset(Dataset):
    def __init__(self, data_dir, img_dir, split='train', img_size=256, clip_model='mobileclip_s0', clip_dir='checkpoints'):
        self.data_path = data_dir
        self.img_dir = img_dir
        self.split = split
        self.img_size = img_size
        self.data = [json.loads(line) for line in open(os.path.join(data_dir, f'{split}.jsonl'), 'r')]

        _, _, preprocess = mobileclip.create_model_and_transforms(clip_model, pretrained=f'checkpoints/{clip_model}.pt')
        self.processor = preprocess
        self.tokenizer = mobileclip.get_tokenizer(clip_model)


    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_name = self.data[idx]['img']
        img_path = os.path.join(self.img_dir, img_name)
        raw_img = Image.open(img_path).convert('RGB').resize((self.img_size, self.img_size))

        raw_text = self.data[idx]['text']
        tokenized_text = self.tokenizer(raw_text)

        preprocessed_img = self.processor(raw_img).unsqueeze(0)
        label = self.data[idx]['label']

        return raw_img, raw_text, preprocessed_img, tokenized_text, label
    
