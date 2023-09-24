import torch 
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple
import numpy as np
import json

from framework.dataset import get_dataset

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_codes,
                 #input_ids,
                 idx,
                 label,

    ):
        self.input_codes = input_codes
        #self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label

class test_source(Dataset):
    def __init__(self, root: str, split: str , preprocess_format, args):
        self.n_classes = 2
        assert split in ['train', 'val', 'test']
        #according to the 'split' to split the dataset
        #split = 'training' if split == 'train' else 'validation'
        print(args)
        from types import SimpleNamespace
        args = SimpleNamespace(**args)
        
        #chose smaller datasets for testing purposes
        if split == 'test':
            file_path = args.eval_data_file
        elif split == 'val':
            file_path = args.train_data_file
        else:
            file_path = args.test_data_file
        
        sample_percent = args.training_percent if args.training_percent else 1.0

        #preprocess
        self.preprocess = preprocess_format

        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                #source
                code=' '.join(js['func'].split())
                #code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
                source_codes = code
                #source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
                #source_ids+=[tokenizer.pad_token_id]*padding_length
                self.examples.append(InputFeatures(source_codes,js['idx'],js['target']))
                #self.examples.append(js)

        total_len = len(self.examples)
        num_keep = int(sample_percent * total_len)

        if num_keep < total_len:
            np.random.seed(10)
            np.random.shuffle(self.examples)
            self.examples = self.examples[:num_keep]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        input_x = self.examples[i].input_codes
        label = torch.tensor(self.examples[i].label)
        #input_x = torch.tensor(self.examples[i].input_ids)
        #label = torch.tensor(self.examples[i].label)

        if self.preprocess:
            #print("former data:")
            #print(input_x.shape)
            #print(label)
            input_x, label = self.preprocess(input_x, label)       
            #print("preprocessed data:")
            #print(input_x.shape)
            #print(label)
            #assert 0 == 1
        return input_x, label
    

if __name__ == "__main__":
    """  parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        ordered_dict = ordered_load(f, yaml.SafeLoader)
        cfg = ordered_dict
    #dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    trainset = get_dataset(cfg , 'train')
    #valset = get_dataset(cfg , 'val') """
    args = {
        'train_data_file' : '../dataset/train.jsonl',
        'eval_data_file' : '../dataset/valid.jsonl',
        'test_data_file' : '../dataset/test.jsonl',
        'training_percent': 1.0
    }
    test_dataset = test_source('dataset', 'train', False, args)
    for idx, (input_x, label) in enumerate(test_dataset):
        print(f"Sample #{idx} - Input: {input_x}, Label: {label}")
        if idx > 10:  # print only first 10 samples
            break
    