import os
import torch
from torch.utils.data import DataLoader, Dataset
import subprocess
from types import SimpleNamespace
from .IVDetectDataset_build import IVD_dataset_build
import random
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#from utils import some_module


class IVDetectDataset(Dataset):
    def __init__(self, split: str, root: str, preprocess_format, args):
        args = SimpleNamespace(**args)
        self.datapoint_files = []
        self.file_dir = ''
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            sub_dirs = ['pyg_graph', 'train_graph', 'test_graph', 'valid_graph']
            for sub_dir in sub_dirs:
                os.makedirs(os.path.join(data_dir, sub_dir))
            IVD_dataset_build(True)
            cmd_path = os.path.join(os.path.dirname(__file__), 'glove', 'ast.sh')
            cmd_1 = subprocess.run(['/bin/sh', cmd_path], cwd=os.path.join(os.path.dirname(__file__), 'glove'))
            cmd_path = os.path.join(os.path.dirname(__file__), 'glove', 'pdg.sh')
            cmd_2 = subprocess.run(['/bin/sh', cmd_path], cwd=os.path.join(os.path.dirname(__file__), 'glove'))
            IVD_dataset_build(False)

            # Call split_dataset
            self.split_dataset(split)
        
        if split == 'train':
            train_path = os.path.join(os.path.dirname(__file__), 'data/train_graph/')
            train_files = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
            self.datapoint_files = train_files
            self.file_dir = train_path

        elif split == 'valid':
            valid_path = os.path.join(os.path.dirname(__file__), 'data/valid_graph/')
            valid_files = [f for f in os.listdir(valid_path) if os.path.isfile(os.path.join(valid_path, f))]
            self.datapoint_files = valid_files
            self.file_dir = valid_path

        elif split == 'test':
            test_path = os.path.join(os.path.dirname(__file__), 'data/test_graph/')
            test_files = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
            self.datapoint_files = test_files
            self.file_dir = test_path
        print('datapoints, file_dir:')
        print(len(self.datapoint_files))
        print(self.file_dir)
        print('Finished Dataset building...')

    def split_dataset(self, split):
        graph_path = os.path.join(os.path.dirname(__file__), 'data/pyg_graph/')
        graph_files = [f for f in os.listdir(graph_path) if os.path.isfile(os.path.join(graph_path, f))]

        vul_list = []
        nonvul_list = []
        for graph_file in tqdm(graph_files):
            graph = torch.load(os.path.join(graph_path, graph_file))
            if graph.y == 0:
                nonvul_list.append(graph_file)
            else:
                vul_list.append(graph_file)
        
        len_vul = len(vul_list)
        len_nonvul = len(nonvul_list)
        ratio = len_nonvul / len_vul
        vul_partition = int(0.6 * len_vul)
        train_list = vul_list[:vul_partition] + nonvul_list[:vul_partition]
        random.shuffle(train_list)
        non_vul_valid = int(int(0.1 * len_vul) * ratio)
        valid_list = vul_list[vul_partition:vul_partition + int(0.1 * len_vul)] + nonvul_list[vul_partition:vul_partition + non_vul_valid]
        random.shuffle(valid_list)
        test_list = vul_list[vul_partition + int(0.1 * len_vul):] + nonvul_list[vul_partition + non_vul_valid:]
        
        print(f"train: {len(train_list)}\tvalid: {len(valid_list)}\ttest: {len(test_list)}")

        if split == 'train':
            self.save_to_files(train_list, 'train_graph')
        elif split == 'val':
            self.save_to_files(valid_list, 'valid_graph')
        elif split == 'test':
            self.save_to_files(test_list, 'test_graph')

    def save_to_files(self, file_list, split_dir):
        for index, graph_file in enumerate(tqdm(file_list)):
            graph = torch.load(os.path.join(os.path.dirname(__file__),'data/pyg_graph', graph_file))
            torch.save(graph, os.path.join(os.path.dirname(__file__),f'data/{split_dir}', f'data_{index}.pt'))

    def __getitem__(self, index):
        graph_file = os.path.join(self.file_dir, self.datapoint_files[index])
        graph = torch.load(graph_file)
        #graph = graph.cuda()
        input_x_data = graph.my_data#torch.tensor(graph.my_data)
        input_x_edge_index = graph.edge_index#torch.tensor(graph.edge_index)
        input_x = (input_x_data, input_x_edge_index)#torch.tensor(graph.my_data, graph.edge_index)
        label = graph.y#torch.tensor(graph.y)
        print(graph)
        #print(graph.my_data.shape)
        #print(graph.edge_index)
        print(label)
        return input_x, label

    def __len__(self):
        return len(self.datapoint_files)
