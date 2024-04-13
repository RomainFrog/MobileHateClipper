import argparse
import os
from pathlib import Path

from datasets import HatefulMemesDataset

def get_args_parser():
    parser = argparse.ArgumentParser('Hateful Memes training/evaluation script', add_help=False)
    pass
    
def main(args):
    pass

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)