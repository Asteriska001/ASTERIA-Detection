import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import subprocess
from .IVDetectDataset_build import IVD_dataset_build

class IVDetectDatset(Dataset):
    def __init__(self, _datapoint_files, file_dir):
        self.datapoint_files = _datapoint_files
        self.file_dir = file_dir
        IVD_dataset_build(False)
        cmd_1 = subprocess.Popen('bash glove/ash.sh')
        cmd_2 = subprocess.Popen('bash glove/pdg.sh')
        IVD_dataset_build(True)
        print('Finished Dataset building...')

    def __getitem__(self, index):
        graph_file = os.getcwd() + "/{}".format(self.file_dir) + "{}".format(self.datapoint_files[index])
        graph = torch.load(graph_file)
        return graph

    def __len__(self):
        return len(self.datapoint_files)