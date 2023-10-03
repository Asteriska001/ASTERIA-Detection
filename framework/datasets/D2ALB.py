import pandas as pd
import torch
from torch.utils.data import Dataset
import requests


class D2ALB_TraceDataset(Dataset):
    def __init__(self, root: str, split: str , tokenizer, preprocess_format, args):
        csv_file = args.csv_file
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'id': self.data.iloc[idx]['id'],
            'trace': self.data.iloc[idx]['trace']
        }
        if 'label' in self.data.columns:
            sample['label'] = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.float32)
        input_x = sample
        return input_x, sample['label']

class D2ALB_CodeDataset(Dataset):
    def __init__(self, root: str, split: str , tokenizer, preprocess_format, args):
        csv_file = args.csv_file
        self.data = pd.read_csv(csv_file)

    def download_bug_url(self, url):
        response = requests.get(url)
        return response.text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'id': self.data.iloc[idx]['id'],
            'bug_url': self.data.iloc[idx]['bug_url'],
            'bug_function': self.data.iloc[idx]['bug_function'],
            'functions': self.data.iloc[idx]['functions']
        }
        if 'label' in self.data.columns:
            sample['label'] = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.float32)
        input_x = sample
        return input_x, sample['label']

class D2ALB_TraceCodeDataset(Dataset):
    def __init__(self, root: str, split: str , tokenizer, preprocess_format, args):
        csv_file = args.csv_file
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'id': self.data.iloc[idx]['id'],
            'bug_url': self.data.iloc[idx]['bug_url'],
            'bug_function': self.data.iloc[idx]['bug_function'],
            'functions': self.data.iloc[idx]['functions'],
            'trace': self.data.iloc[idx]['trace']
        }
        if 'label' in self.data.columns:
            sample['label'] = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.float32)
        input_x = sample
        return input_x, sample['label']


class D2ALB_FunctionDataset(Dataset):
    def __init__(self, root: str, split: str , tokenizer, preprocess_format, args):
        csv_file = args.csv_file
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'id': self.data.iloc[idx]['id'],
            'code': self.data.iloc[idx]['code']
        }
        if 'label' in self.data.columns:
            sample['label'] = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.float32)
        input_x = sample
        return input_x, sample['label']
