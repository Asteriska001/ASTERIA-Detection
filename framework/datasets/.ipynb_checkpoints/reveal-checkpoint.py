import torch 
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple
import numpy as np
import json

from framework.models.modules.transformers.transformers import *

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label


def convert_examples_to_features(js,tokenizer,args):
    #source
    code=' '.join(js['func'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['idx'],js['target'])


class REVEAL(Dataset):
    def __init__(self, root: str, split: str , tokenizer, preprocess_format, args):
            #port dataset
            assert split in ['train', 'val']
            #according to the 'split' to split the dataset
            #split = 'training' if split == 'train' else 'validation'
            print(args)
            from types import SimpleNamespace
            args = SimpleNamespace(**args)
            
            if split == 'val':
                file_path = args.eval_data_file
            elif split == 'train':
                file_path = args.train_data_file
            else:
                file_path = args.test_data_file
            
            sample_percent = args.training_percent if args.training_percent else 1.0

            config_class, model_class, tokenizer_class = MODEL_CLASSES[tokenizer]
            tokenizer = tokenizer_class.from_pretrained('microsoft/graphcodebert-base',
                                                do_lower_case = None,#args.do_lower_case,
                                                       )#cache_dir=args.cache_dir if args.cache_dir else None)
            
            #preprocess
            self.preprocess = preprocess_format

            self.examples = []
            with open(file_path) as f:
                for line in f:
                    js=json.loads(line.strip())
                    self.examples.append(convert_examples_to_features(js, tokenizer, args))

            total_len = len(self.examples)
            num_keep = int(sample_percent * total_len)

            if num_keep < total_len:
                np.random.seed(10)
                np.random.shuffle(self.examples)
                self.examples = self.examples[:num_keep]
                '''logger 待处理
            if 'train' in file_path:
                logger.info("*** Total Sample ***")
                logger.info("\tTotal: {}\tselected: {}\tpercent: {}\t".format(total_len, num_keep, sample_percent))
                for idx, example in enumerate(self.examples[:3]):
                        logger.info("*** Sample ***")
                        logger.info("Total sample".format(idx))
                        logger.info("idx: {}".format(idx))
                        logger.info("label: {}".format(example.label))
                        logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                        logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                '''
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        input_x = torch.tensor(self.examples[i].input_ids)
        label = torch.tensor(self.examples[i].label)

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
