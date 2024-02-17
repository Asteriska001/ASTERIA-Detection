# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
from framework.models.modules.transformers.transformers import *

    
    
class CodeXGLUE_baseline(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(CodeXGLUE_baseline, self).__init__()
        from types import SimpleNamespace
        args = SimpleNamespace(**args)
        config_class, model_class, tokenizer_class = MODEL_CLASSES[tokenizer]
        
        encoder = model_class.from_pretrained('microsoft/codebert-base',#args.model_name_or_path,
                                            #from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                             )#cache_dir=args.cache_dir if args.cache_dir else None)
        self.encoder = encoder
        
        tokenizer = tokenizer_class.from_pretrained('microsoft/codebert-base',
                                                do_lower_case = None,#args.do_lower_case,
                                                       )#cache_dir=args.cache_dir if args.cache_dir else None)
        self.tokenizer = tokenizer
        
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                             )#cache_dir=args.cache_dir if args.cache_dir else None)
        
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(0)

        
    def forward(self, input_ids=None):#,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]

        # Apply dropout
        outputs = self.dropout(outputs)

        logits=outputs
        prob=torch.sigmoid(logits)
        return prob
        # if labels is not None:
        #     labels=labels.float()
        #     loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
        #     loss=-loss.mean()
        #     return loss,prob
        # else:
        #     return prob
      
        
 