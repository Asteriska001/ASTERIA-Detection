import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification
from framework.models.modules.transformers.transformers import *

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class LineVul(RobertaForSequenceClassification):   
    def __init__(self, encoder, config, tokenizer, args):
        
        from types import SimpleNamespace
        args = SimpleNamespace(**args)
        self.args = args
        
        config_class, model_class, tokenizer_class = MODEL_CLASSES[tokenizer]
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                             )#cache_dir=args.cache_dir if args.cache_dir else None)
        config.num_labels = 1
        super(LineVul, self).__init__(config=config)
                
        encoder = model_class.from_pretrained('microsoft/graphcodebert-base',#args.model_name_or_path,
                                            #from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                             )#cache_dir=args.cache_dir if args.cache_dir else None)
        tokenizer = tokenizer_class.from_pretrained('microsoft/graphcodebert-base',
                                                do_lower_case = None,#args.do_lower_case,
                                                       )#cache_dir=args.cache_dir if args.cache_dir else None)
        
        
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
    
        
    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None):
        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            else:
                input_ids = input_embed.long()
                outputs = self.encoder.roberta(input_ids=input_ids, output_attentions=output_attentions)[0]
            '''
            if input_ids is not None:
                #need fix.
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            else:
                input_ids = input_embed.long()
                outputs = self.encoder.roberta(input_ids=input_ids, output_attentions=output_attentions)
                #outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)#[0]
            '''
            logits = self.classifier(outputs)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                return prob