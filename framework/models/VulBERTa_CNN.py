import torch

class VulBERTa_CNN(torch.nn.Module):
    def __init__(self,base_model,n_classes,base_model_output_size=768, dropout=0.2):
        super().__init__()
        
        self.num_labels = n_classes
        self.base_model = base_model
        self.dropout1 = torch.nn.Dropout(dropout)
        #self.dropout2 = torch.nn.Dropout(dropout)
        #self.fc1 = torch.nn.Linear(768,512)
        self.fc2 = torch.nn.Linear(300,128)
        self.fc3 = torch.nn.Linear(128,n_classes)
        
#        self.conv = torch.nn.Conv1d(in_channels=768, out_channels=512, kernel_size=9)
        
        self.conv1 = torch.nn.Conv1d(in_channels=768, out_channels=100, kernel_size=3)
        self.conv2 = torch.nn.Conv1d(in_channels=768, out_channels=100, kernel_size=4)
        self.conv3 = torch.nn.Conv1d(in_channels=768, out_channels=100, kernel_size=5)
        
    def forward(self,input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,output_attentions=None,output_hidden_states=None,return_dict=None):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        
        x = outputs[0]
        #x = x[:, 0, :]
        x = x.permute(0,2,1)
        x1 = torch.nn.functional.relu(self.conv1(x))
        x2 = torch.nn.functional.relu(self.conv2(x))
        x3 = torch.nn.functional.relu(self.conv3(x))
        
        x1 = torch.nn.functional.max_pool1d(x1, x1.shape[2])
        x2 = torch.nn.functional.max_pool1d(x2, x2.shape[2])
        x3 = torch.nn.functional.max_pool1d(x3, x3.shape[2])
        
        x = torch.cat([x1,x2,x3],dim=1)
        x = x.flatten(1)
        
#         x = torch.nn.functional.relu(self.conv(x))
#         x = torch.nn.functional.max_pool1d(x, 4)
#         x = torch.mean(x, -1)
#         x = self.dropout1(x)
        
        
        x = self.fc2(x)
        logits = self.fc3(x)
        
        #### Below is the standard output from RobertaforSequenceClassifcation head class
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

