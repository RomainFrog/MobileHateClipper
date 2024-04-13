import argparse
import os
import torch
import numpy as np
from pathlib import Path

from datasets import get_hateful_memes_dataset

def get_args_parser():
    parser = argparse.ArgumentParser('Hateful Memes training/evaluation script', add_help=False)

    # dataset parameters
    parser.add_argument('--data_dir', default='hateful_memes/', type=Path, help='path to data directory')
    parser.add_argument('--img_dir', default='hateful_memes/', type=Path, help='path to image directory')
    parser.add_argument('--output_dir', default='./output_dir', type=Path, help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int, help='seed for training')
    parser.add_argument('--shuffle', default=True, type=bool, help='shuffle dataset')

    # model parameters
    parser.add_argument('--clip_model', default='mobileclip_s0', type=str, help='CLIP model name')
    parser.add_argument('--clip_checkpoint', default='checkpoints', type=Path, help='path to CLIP model checkpoints')
    parser.add_argument('--img_size', default=244, type=int, help='image size')

    # training parameters
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')

    return parser
    
def main(args):

    print('job dir: {}\n'.format(os.getcwd()))
    print("{}".format(args).replace(', ', ',\n'))
    print()

    device = torch.device(args.device)

    # fix random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_train = get_hateful_memes_dataset(args=args, split='train')
    dataset_val = get_hateful_memes_dataset(args=args, split='dev_seen')

    print(f"Number of training examples: {len(dataset_train)}")
    print(f"Number of validation examples: {len(dataset_val)}")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Number of training batches: {len(data_loader_train)} (batch size: {args.batch_size})")
    print(f"Number of validation batches: {len(data_loader_val)} (batch size: {args.batch_size})")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)