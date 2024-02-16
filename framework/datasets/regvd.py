import torch 
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple
import numpy as np
import json,math

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

        
#draper csv
def convert_examples_to_features_draper(js,tokenizer,args):
    #source
    #print(js)
    try:
        code=' '.join(js['func'].split())
        code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        return InputFeatures(source_tokens,source_ids,js['idx'],js['target'])
    except:
        print(js)
#reveal csv
def convert_examples_to_features_reveal(js,tokenizer,args):
    #source
    #print(js)
    try:
        code=' '.join(js['code'].split())
        code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        return InputFeatures(source_tokens,source_ids,js['idx'],js['target'])
    except:
        print(js)
#diverse csv
def convert_examples_to_features_diverse(js,tokenizer,args):
    #source
    #print(js)
    try:
        code=' '.join(js['func'].split())
        code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        return InputFeatures(source_tokens,source_ids,js['idx'],js['target'])
    except:
        print(js)
#d2a csv
def convert_examples_to_features_d2a(js,tokenizer,args):
    #source
    code=' '.join(js['code'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['id'],js['label'])

#MSR csv
def convert_examples_to_features_MSR(js,tokenizer,args):
    #source
    code=' '.join(js['processed_func'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['index'],js['target'])
        
def convert_examples_to_features(js,tokenizer,args):
    #source
    code=' '.join(js['func'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['idx'],js['target'])
        
# def convert_examples_to_features(js,tokenizer,args):
#     #source
#     code=' '.join(js['code'].split())
#     code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
#     source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
#     source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
#     padding_length = args.block_size - len(source_ids)
#     source_ids+=[tokenizer.pad_token_id]*padding_length
#     return InputFeatures(source_tokens,source_ids,js['size'],1)

def convert_examples_to_features_csv(js,tokenizer,args):
    #vdet
    #print('+'+js)
    code=' '.join(js['Code'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    target = int(js['isVulnerable'])
    return InputFeatures(source_tokens,source_ids,js['id'],target)


class ReGVD(Dataset):
    def __init__(self, root: str, split: str , tokenizer, preprocess_format, args):
            self.n_classes = 2
            #port dataset
            assert split in ['train', 'val', 'test']
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
            # reveal original data
            # with open(file_path) as f:
            #     datas = json.loads(f.read())                
            #     for js in datas:
            #         self.examples.append(convert_examples_to_features(js, tokenizer, args))
            
            #vdet csv data
            # import pandas as pd
            # df = pd.read_csv(file_path)
            # for idx, record in df.iterrows():
            #     self.examples.append(convert_examples_to_features_d2a(record, tokenizer, args))
            
            # regvd original data
            # with open(file_path) as f:
            #     for line in f:
            #         js=json.loads(line.strip())
            #         self.examples.append(convert_examples_to_features(js, tokenizer, args))
            import pandas as pd
            df = pd.read_excel(file_path)
            for idx, record in df.iterrows():
                self.examples.append(convert_examples_to_features_draper(record, tokenizer, args))
            
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
        try:
            # Attempt to convert label to float
            label_value = float(self.examples[i].label)
            
            # Check for NaN and handle it
            if math.isnan(label_value):
                # Handle NaN value, set to default value, e.g., 0
                label = torch.tensor(0)
            else:
                # Convert to integer and then tensor
                label = torch.tensor(int(label_value))
        except ValueError:
            # Handle case where conversion to float fails
            # Set to a default value or handle differently
            label = torch.tensor(0)  # Example: default value
        #label = torch.tensor(int(self.examples[i].label))

        if self.preprocess:
            #print("former data:")
            #print(input_x.shape)
            #print(label)
            input_x, label = self.preprocess(input_x, label)       
            #print("preprocessed data:")
            #print(input_x.shape)
            #print(label)
            #assert 0 == 1
        # print('input_x: ',input_x)
        # print('label: ',label)
        return input_x, label
