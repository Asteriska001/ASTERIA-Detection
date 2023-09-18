import torch
from torch import nn

def get_pretrained_model(pretrained_config):
    if pretrained_config:
        return torch.load_state_dict(torch.load(pretrained_config, map_location='cpu'), strict=False)