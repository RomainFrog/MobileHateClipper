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
    parser.add_argument('--data_dir', default='data/hateful_memes/', type=Path, help='path to data directory')
    parser.add_argument('--img_dir', default='data/hateful_memes/', type=Path, help='path to image directory')
    parser.add_argument('--output_dir', default='./experiments/output_dir', type=Path, help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int, help='seed for training')
    parser.add_argument('--shuffle', default=True, type=bool, help='shuffle dataset')
    parser.add_argument('--eval', action='store_true', help='evaluation only')
    parser.add_argument('--use_propaganda', action='store_true', help='use propaganda labels')
    parser.add_argument('--use_memotion', action='store_true', help='use memotion labels')

    # model parameters
    parser.add_argument('--clip_model', default='mobileclip_s0', type=str, help='CLIP model name')
    parser.add_argument('--clip_checkpoint', default='checkpoints', type=Path, help='path to CLIP model checkpoints')
    parser.add_argument('--img_size', default=256, type=int, help='image size')
    parser.add_argument('--fusion', default='align', type=str, help='fusion method')
    parser.add_argument('--num_pre_output_layers', default=2, type=int, help='number of pre-output layers')
    parser.add_argument('--num_mapping_layers', default=1, type=int, help='number of mapping layers')
    parser.add_argument('--embed_dim', default=512, type=int, help='embedding dimension')
    parser.add_argument('--pre_output_dim', default=512, type=int, help='pre-output dimension')
    parser.add_argument('--dropouts', default=[0.2, 0.4, 0.2], type=list, help='dropouts for projection, fusion and pre-output layers')
    parser.add_argument('--freeze_clip', default=True, type=bool, help='freeze the clip encoder')

    # training parameters
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--finetune', default='', type=str, help='finetune from checkpoint')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--clip_grad_norm', default=0.1, type=float, help='clip grad norm')
    parser.add_argument('--epochs', default=20, type=int, help='number of training epochs')
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

    # dataset_train = get_hateful_memes_dataset(args=args, split='train', strategy='KFold', fold=4, n_splits=5)
    # dataset_val = get_hateful_memes_dataset(args=args, split='val', strategy='KFold', fold=4, n_splits=5)
    dataset_train = get_hateful_memes_dataset(args=args, split='train')
    dataset_val = get_hateful_memes_dataset(args=args, split='dev_seen')


    print(f"Number of training examples: {len(dataset_train)}")
    print(f"Number of validation examples: {len(dataset_val)}")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        # drop_last=True,
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

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), amsgrad=False)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    print(f'Optimize: {optimizer}')
    print(f'Criterion: {criterion}')


    # auxilliary variables
    best_AUROC = 0.5
    best_model = None
    best_epoch = 0
    if not args.eval:
    # training loop
        print(f"\nStart training for {args.epochs} epochs")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):

            train_stats = train_one_epoch(
                model=model, criterion=criterion, 
                data_loader=data_loader_train, 
                optimizer=optimizer, device=device, 
                epoch=epoch, max_grad_norm=args.clip_grad_norm)
            
            val_stats = evaluate(model, data_loader_val, device, criterion)
            print(f"Epoch {epoch} | Train Loss: {train_stats['loss']:.4f} | Val Loss: {val_stats['loss']:.4f}")
            print(f'Accuracy: {val_stats["accuracy"]:.4f} | F1: {val_stats["f1"]:.4f} | AUROC: {val_stats["auroc"]:.4f}\n')

            # save the best model
            if val_stats["auroc"] > best_AUROC:
                best_AUROC = val_stats["auroc"]
                best_model = model.state_dict()
                best_epoch = epoch

            if args.output_dir:
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                # compute time elapsed
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                with open(os.path.join(args.output_dir, 'log.jsonl'), 'a') as f:
                    f.write(f'{{"time": {total_time_str} ,"train_loss": {train_stats["loss"]:.4f}, "val_loss": {val_stats["loss"]:.4f}, "val_acc": {val_stats["accuracy"]:.4f}, "val_f1": {val_stats["f1"]:.4f}, "val_auroc": {val_stats["auroc"]:.4f}, "epoch": {epoch}}}\n')
            
            # if args.output_dir:
            #     torch.save(model.state_dict(), os.path.join(args.output_dir, f'checkpoint-{epoch}.pth'))
            

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
        # save the best model
        if best_model:
            if args.output_dir:
                torch.save(best_model, os.path.join(args.output_dir, f'checkpoint-best-{best_epoch}.pth'))
    
    else:
        dataset_test = get_hateful_memes_dataset(args=args, split='test_seen')
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        print(f"\nStart evaluation")
        val_stats = evaluate(model, data_loader_test, device, criterion)
        print(f'Accuracy: {val_stats["accuracy"]:.4f} | F1: {val_stats["f1"]:.4f} | AUROC: {val_stats["auroc"]:.4f}\n')
        # save stats to val.jsonl
        if args.output_dir:
            with open(os.path.join(args.output_dir, 'test.jsonl'), 'w') as f:
                f.write(f'{{"accuracy": {val_stats["accuracy"]:.4f}, "f1": {val_stats["f1"]:.4f}, "auroc": {val_stats["auroc"]:.4f}}}\n')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
