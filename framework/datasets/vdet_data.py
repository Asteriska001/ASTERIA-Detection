import torch 
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Tuple
import numpy as np
import pandas as pd
import json
import random
import transformers

#torch.multiprocessing.set_start_method('spawn')

torch.cuda.empty_cache()

def tokenize_truncate(tokenizer, text_samples, max_length):
    full_input_ids = []

    # For each training example...
    for text in text_samples:
        # Tokenize the sample.
        input_ids = tokenizer.encode(text=text,              # Text to encode.
                                    add_special_tokens=True, # Do add specials.
                                    max_length=max_length,      # Do Truncate!
                                    truncation=True,         # Do Truncate!
                                    padding=False)           # DO NOT pad.
                                    
        # Add the tokenized result to our list.
        full_input_ids.append(input_ids)
        
    print('DONE. {:>10,} samples\n'.format(len(full_input_ids)))
    return full_input_ids


def build_batches(samples, batch_size):
    # List of batches that we'll construct.
    batch_ordered_text = []
    batch_ordered_labels = []

    print('Creating batches of size {:}...'.format(batch_size))

    # Loop over all of the input samples...    
    while len(samples) > 0:
        # `to_take` is our actual batch size. It will be `batch_size` until 
        # we get to the last batch, which may be smaller. 
        to_take = min(batch_size, len(samples))

        # Pick a random index in the list of remaining samples to start
        # our batch at.
        select = random.randint(0, len(samples) - to_take)

        # Select a contiguous batch of samples starting at `select`.
        batch = samples[select:(select + to_take)]

        #print("Batch length:", len(batch))

        # Each sample is a tuple--split them apart to create a separate list of 
        # sequences and a list of labels for this batch.
        batch_ordered_text.append([s[0] for s in batch])
        batch_ordered_labels.append([s[1] for s in batch])

        # Remove these samples from the list.
        del samples[select:select + to_take]

    print('\t  DONE - Selected {:,} batches.\n'.format(len(batch_ordered_text)))
    return batch_ordered_text, batch_ordered_labels


def add_padding_per_batch(tokenizer, batch_ordered_text, batch_ordered_labels):
    print('Padding out sequences within each batch...')

    final_input_ids = []
    final_attention_masks = []
    final_labels = []

    # For each batch...
    for (batch_inputs, batch_labels) in zip(batch_ordered_text, batch_ordered_labels):

        # New version of the batch, this time with padded sequences and now with
        # attention masks defined.
        batch_padded_inputs = []
        batch_attn_masks = []
        
        # First, find the longest sample in the batch. 
        # Note that the sequences do currently include the special tokens!
        max_size = max([len(sen) for sen in batch_inputs])

        # For each input in this batch...
        for sen in batch_inputs:
            
            # How many pad tokens do we need to add?
            num_pads = max_size - len(sen)

            # Add `num_pads` padding tokens to the end of the sequence.
            padded_input = sen + [tokenizer.pad_token_id]*num_pads

            # Define the attention mask--it's just a `1` for every real token
            # and a `0` for every padding token.
            attn_mask = [1] * len(sen) + [0] * num_pads

            # Add the padded results to the batch.
            batch_padded_inputs.append(padded_input)
            batch_attn_masks.append(attn_mask)

        # Our batch has been padded, so we need to save this updated batch.
        # We also need the inputs to be PyTorch tensors, so we'll do that here.
        # Todo - Michael's code specified "dtype=torch.long"
        final_input_ids.append(torch.tensor(batch_padded_inputs))
        final_attention_masks.append(torch.tensor(batch_attn_masks))
        final_labels.append(torch.tensor(np.array(batch_labels))) # if there's problems, remove np.array()

    print('\t DONE. Returning final smart-batched data.')
    # Return the smart-batched dataset!
    return (final_input_ids, final_attention_masks, final_labels)


def smart_batching(tokenizer, max_length, text_samples, labels, batch_size):
    # Tokenize and truncate text_samples; no padding
    full_input_ids = tokenize_truncate(tokenizer, text_samples, max_length)

    # Sort the two lists together by the length of the input sequence.
    samples = sorted(zip(full_input_ids, labels), key=lambda x: len(x[0]))

    # Build batches of contiguous data, starting at random points in samples
    batch_size = batch_size
    batch_ordered_text, batch_ordered_labels = build_batches(samples, batch_size)
   
    # Add padding accordingly to batch size
    final_input_ids, final_attention_masks, final_labels = add_padding_per_batch(tokenizer, batch_ordered_text, batch_ordered_labels)

    return final_input_ids, final_attention_masks, final_labels


def tokenize_and_pad(tokenizer, text_samples, max_length):
    input_ids = []
    attention_masks = []

    for text in text_samples:
        encoded_dict = tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

class vdet_data(Dataset):
    def __init__(self, root: str, split: str , tokenizer, preprocess_format, args) -> None:
        
        from types import SimpleNamespace
        args = SimpleNamespace(**args)
        tokenizer = transformers.AutoTokenizer.from_pretrained("CAUKiel/JavaBERT") 
        
        train_dataset = pd.read_csv(args.train_file)
        val_dataset = pd.read_csv(args.eval_file)
        # string to arrays
        from ast import literal_eval
        train_dataset['one-hot'] = train_dataset['one-hot'].apply(literal_eval)
        #val_dataset['one-hot'] = val_dataset['one-hot'].apply(literal_eval)
        
        self.tokenizer = tokenizer
        self.max_length = 512
        self.train_dataset = train_dataset
        self.input_ids, self.attention_masks = tokenize_and_pad(
            tokenizer, 
            self.train_dataset['Code'].tolist(), 
            self.max_length
        )

        self.labels = torch.tensor(self.train_dataset['one-hot'].tolist())
        # max_length = 512
        # text_samples = train_dataset['Code']
        # labels = train_dataset['one-hot']
        # full_input_ids = tokenize_truncate(tokenizer, text_samples, max_length)
        
        #train_input_ids, train_attn_masks, train_labels = smart_batching(tokenizer, 512, train_dataset['Code'], train_dataset['one-hot'], BATCH_SIZE) 
        # test_input_ids, test_attn_masks, test_labels = smart_batching(tokenizer, 512, test_dataset['Code'], test_dataset['one-hot'], BATCH_SIZE)
        #val_input_ids, val_attn_masks, val_labels = smart_batching(tokenizer, 512, val_dataset['Code'], val_dataset['one-hot'], BATCH_SIZE)
        
        #self.examples_input_ids, self.examples_attn_masks, self.examples_labels = train_input_ids, train_attn_masks, train_labels
        
    def __len__(self):
        print('The length of dataset: ',len(self.input_ids))
        return len(self.input_ids)#len(self.examples_input_ids)
    
    def __getitem__(self, index):
        ids = self.input_ids[index]#.to('cuda')#, dtype = torch.long)
        #print('ids shape: ',ids.shape)
        mask = self.attention_masks[index]#.to('cuda')#, dtype = torch.long)
        #print('mask shape: ',mask.shape)
        label = self.labels[index]#.to('cuda')#, dtype = torch.float)
        #print('label shape: ',label.shape)
        return (ids,mask),label