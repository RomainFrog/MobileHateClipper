import argparse
import os
import time
import datetime
import torch
import numpy as np
from pathlib import Path

from datasets import get_hateful_memes_dataset
from models_mobile_hate_clipper import create_model
from engine import train_one_epoch, evaluate

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
    parser.add_argument('--fusion', default='align', type=str, help='fusion method')
    parser.add_argument('--num_pre_output_layers', default=2, type=int, help='number of pre-output layers')

    # training parameters
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--finetune', default='', type=str, help='finetune from checkpoint')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=10, type=int, help='number of training epochs')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')

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

    model = create_model(args)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print(f"\nLoading model from {args.finetune}")
        model.load_state_dict(checkpoint)

    model.to(device)        

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Model = %s" % str(model))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]}
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    print(f"Optimizer: {optimizer}")
    print(f"Criterion: {criterion}")

    # training loop
    print(f"\nStart training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, criterion)
        evaluate(model, data_loader_val, device, criterion)
        end_time = time.time()
        print(f"Epoch {epoch} took {end_time - start_time} seconds")
        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # save model
    if args.output_dir:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pth'))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)