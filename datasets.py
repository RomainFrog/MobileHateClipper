import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
import mobileclip
from sklearn.model_selection import StratifiedKFold

class HatefulMemesDataset(Dataset):
    def __init__(self, data_dir, img_dir, split='train', img_size=256, clip_model='mobileclip_s0', clip_dir='checkpoints',
                 strategy='', fold=0, n_splits=5):
        self.data_path = data_dir
        self.img_dir = img_dir
        self.split = split
        self.img_size = img_size

        if strategy == 'KFold':
            if split not in ['train', 'val']:
                raise ValueError(f"split has to be either 'train' or 'val' for KFold strategy")
            self.data = get_K_fold(data_dir, n_splits=n_splits, split_idx=fold, subset=split)
        else:
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


def get_K_fold(data_dir, n_splits, split_idx, subset: str = "train"):
    """
    Creates a dataset from train, dev_seen and dev_unseen splits
    removes duplicates from the dataset
    Creates a KFold object with n_splits. The split_idx is used to determine the fold to return
    and the subset determines if its the train or validation set of the fold
    """

    data = []
    for split in ["train", "dev_seen", "dev_unseen"]:
        data += [json.loads(line) for line in open(os.path.join(data_dir, f'{split}.jsonl'), 'r')]
    # remove duplicates
    data = list({json.dumps(item, sort_keys=True) for item in data})
    data = [json.loads(item) for item in data]

    # create a stratified KFold object
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(kfold.split(data, [item['label'] for item in data]))

    train_idx, val_idx = splits[split_idx]

    if subset == "train":
        data = [data[idx] for idx in train_idx]
    elif subset == "val":
        data = [data[idx] for idx in val_idx]

    return data



    

def get_hateful_memes_dataset(args, split='train', strategy='', fold=0, n_splits=5):
    return HatefulMemesDataset(
        data_dir=args.data_dir,
        img_dir=args.img_dir,
        split=split,
        img_size=args.img_size,
        clip_model=args.clip_model,
        clip_dir=args.clip_checkpoint,
        strategy=strategy,
        fold=fold,
        n_splits=n_splits
        )
