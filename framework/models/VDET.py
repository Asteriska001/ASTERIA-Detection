import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
#from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import numpy as np
import random


#tokenizer = transformers.AutoTokenizer.from_pretrained("CAUKiel/JavaBERT") 

class vdet_for_java(nn.Module):
    DROPOUT_PROB = 0.1 # default value
    N_CLASSES = 23 

    def __init__(self,encoder, config, tokenizer, args):
        super(vdet_for_java, self).__init__()
        self.model = transformers.AutoModel.from_pretrained("CAUKiel/JavaBERT", output_hidden_states=True) 
        self.dropout = nn.Dropout(self.DROPOUT_PROB) 
        self.linear = nn.Linear(768 * 4, self.N_CLASSES) # If you are using last four hidden state
        # self.linear = nn.Linear(768, self.N_CLASSES) # If you are using the pooler output
        self.step_scheduler_after = "batch"


    def forward(self, input_x):
        """Use last four hidden states"""
        ids, mask = input_x
        ids = ids.to('cuda')
        mask = mask.to('cuda')
        all_hidden_states = torch.stack(self.model(ids, attention_mask=mask)["hidden_states"])

        concatenate_pooling = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]),-1
        )

        concatenate_pooling = concatenate_pooling[:, 0]

        output_dropout = self.dropout(concatenate_pooling)
        
        output = self.linear(output_dropout)
        #print('model outputs and shape: ',output,output.shape)
        return output

    # def forward(self, ids, mask):
    #     """Use pooler output"""
    #     output_1 = self.model(ids, attention_mask=mask)["pooler_output"]
    #     output_dropout = self.dropout(output_1)
    #     output = self.linear(output_dropout)
    #     return output 