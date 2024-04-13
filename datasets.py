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
        checkpoint_path = os.path.join(clip_dir, f'{clip_model}.pt')
        _, _, preprocess = mobileclip.create_model_and_transforms(clip_model, pretrained=checkpoint_path)
        self.processor = preprocess
        self.tokenizer = mobileclip.get_tokenizer(clip_model)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data[idx]['img']
        img_path = os.path.join(self.img_dir, img_name)
        raw_img = Image.open(img_path).convert('RGB').resize((self.img_size, self.img_size))

        raw_text = self.data[idx]['text']
        tokenized_text = self.tokenizer(raw_text).squeeze(0) # (sequlen,)

        preprocessed_img = self.processor(raw_img) # (3, img_size, img_size)
        label = self.data[idx]['label']

        return preprocessed_img, tokenized_text, label
    

def get_hateful_memes_dataset(args, split='train'):
    return HatefulMemesDataset(
        data_dir=args.data_dir,
        img_dir=args.img_dir,
        split=split,
        img_size=args.img_size,
        clip_model=args.clip_model,
        clip_dir=args.clip_checkpoint
        )