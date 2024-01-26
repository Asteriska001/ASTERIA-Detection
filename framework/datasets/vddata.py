import torch 
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple
import numpy as np
import json

import os
import pandas
from .vddata_utils.clean_gadget import clean_gadget
from .vddata_utils.vectorize_gadget import GadgetVectorizer

from framework.models.modules.transformers.transformers import *


"""
Parses gadget file to find individual gadgets
Yields each gadget as list of strings, where each element is code line
Has to ignore first line of each gadget, which starts as integer+space
At the end of each code gadget is binary value
    This indicates whether or not there is vulnerability in that gadget
"""
def parse_file(filename):
    with open(filename, "r", encoding="utf8") as file:
        gadget = []
        gadget_val = 0
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            if "-" * 33 in line and gadget: 
                yield clean_gadget(gadget), gadget_val
                gadget = []
            elif stripped.split()[0].isdigit():
                if gadget:
                    # Code line could start with number (somehow)
                    if stripped.isdigit():
                        gadget_val = int(stripped)
                    else:
                        gadget.append(stripped)
            else:
                gadget.append(stripped)

"""
Uses gadget file parser to get gadgets and vulnerability indicators
Assuming all gadgets can fit in memory, build list of gadget dictionaries
    Dictionary contains gadgets and vulnerability indicator
    Add each gadget to GadgetVectorizer
Train GadgetVectorizer model, prepare for vectorization
Loop again through list of gadgets
    Vectorize each gadget and put vector into new list
Convert list of dictionaries to dataframe when all gadgets are processed
"""
def get_vectors_df(filename, vector_length=100):
    gadgets = []
    count = 0
    vectorizer = GadgetVectorizer(vector_length)
    for gadget, val in parse_file(filename):
        count += 1
        print("Collecting gadgets...", count, end="\r")
        vectorizer.add_gadget(gadget)
        row = {"gadget" : gadget, "val" : val}
        gadgets.append(row)
    print('Found {} forward slices and {} backward slices'
          .format(vectorizer.forward_slices, vectorizer.backward_slices))
    print()
    print("Training model...", end="\r")
    vectorizer.train_model()
    print()
    vectors = []
    count = 0
    for gadget in gadgets:
        count += 1
        print("Processing gadgets...", count, end="\r")
        vector = vectorizer.vectorize(gadget["gadget"])
        row = {"vector" : vector, "val" : gadget["val"]}
        vectors.append(row)
    print()
    df = pandas.DataFrame(vectors)
    return df

class VDdata(Dataset):
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
            
            parse_file(file_path)
            base = os.path.splitext(os.path.basename(file_path))[0]
            vector_filename = base + "_gadget_vectors.pkl"
            vector_length = 50
            if os.path.exists(vector_filename):
                df = pandas.read_pickle(vector_filename)
            else:
                df = get_vectors_df(file_path, vector_length)
                df.to_pickle(vector_filename)
            self.dataset = df
            #print('self dataset:\n',self.dataset)
            
            sample_percent = args.training_percent if args.training_percent else 1.0

            config_class, model_class, tokenizer_class = MODEL_CLASSES[tokenizer]
            tokenizer = tokenizer_class.from_pretrained('microsoft/graphcodebert-base',
                                                do_lower_case = None,#args.do_lower_case,
                                                       )#cache_dir=args.cache_dir if args.cache_dir else None)
            
            #preprocess
            self.preprocess = preprocess_format

            # self.examples = []
            # # with open(file_path) as f:
            # #     for line in f:
            # #         js=json.loads(line.strip())
            # #         self.examples.append(convert_examples_to_features(js, tokenizer, args))

            # total_len = len(self.examples)
            # num_keep = int(sample_percent * total_len)

            # if num_keep < total_len:
            #     np.random.seed(10)
            #     np.random.shuffle(self.examples)
            #     self.examples = self.examples[:num_keep]
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
        print('length of dataset:'+str(len(self.dataset)))
        return len(self.dataset)

    def __getitem__(self, index):
        #print('_getitem_',self.dataset.iloc[index])
        input_x = self.dataset.iloc[index].vector
        # print('inputx shape:',input_x.shape)
        #input_x = torch.tensor(self.dataset.iloc[index].input)
        label = torch.tensor(self.dataset.iloc[index].val)
        return input_x, label
        # input_x = torch.tensor(self.examples[i].input_ids)
        # label = torch.tensor(self.examples[i].label)

        # if self.preprocess:
        #     #print("former data:")
        #     #print(input_x.shape)
        #     #print(label)
        #     input_x, label = self.preprocess(input_x, label)       
        #     #print("preprocessed data:")
        #     #print(input_x.shape)
        #     #print(label)
        #     #assert 0 == 1
        # return input_x, label