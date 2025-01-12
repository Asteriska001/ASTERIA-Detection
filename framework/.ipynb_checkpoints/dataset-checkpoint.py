import torch
from torch import nn
#from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist

from framework.datasets import *
from framework.datasets.XFGDataset_build import DWK_Dataset
from framework.preprocess import get_preprocess

from torch_geometric.data import Data,DataLoader
from typing import List


#手动构建字典
DATASETS_DICT = {
    'ReGVD': ReGVD,
    'Devign_Partial': Devign_Partial,
    'CodeXGLUE': CodeXGLUE,
    'DWK_Dataset': DWK_Dataset,
    'IVDDataset': IVDetectDataset,
    'LineVul': LineVul,
    'VDdata': VDdata,
    'vdet_data': vdet_data
}


# 自动构建字典
#DATASETS_DICT = {k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, nn.Module)}


def get_dataset(config, split):

    dataset_cfg = config['DATASET']
    train_cfg, eval_cfg = config['TRAIN'], config['EVAL']

    dataset_name = dataset_cfg['NAME']  # 这个应该是从配置文件中读取的模型名称
    dataset_param = dataset_cfg['PARAMS']  # 这个应该是从配置文件中读取的模型参数
    #待测试 不知道能不能直接把model——param的字典转为参数传递
    # print(dataset_name)
    # print(dataset_param)
    # print(DATASETS_DICT)
    if dataset_name in DATASETS_DICT:
        
        #preprocess dataset
        preprocess_cfg = dataset_cfg['PREPROCESS']
        preprocess_format = None
        if preprocess_cfg['ENABLE']:
            #get preprocess compose 
            preprocess_compose = preprocess_cfg['COMPOSE']
            print('preprocess compose:')
            print(preprocess_compose)

            if split == 'train':
                preprocess_format = get_preprocess(train_cfg['INPUT_SIZE'], preprocess_compose)
            elif split == 'val':
                preprocess_format = get_preprocess(eval_cfg['INPUT_SIZE'], preprocess_compose)
            else:
                print(split)
                print('Invalid split specified')

        #construct dataset
        dataset = DATASETS_DICT[dataset_name](split = split, 
                                              root = dataset_cfg['ROOT'],
                                              preprocess_format = preprocess_format, 
                                              **dataset_param
                                              )
    else:
        print("The dataset name {} does not exist".format(dataset_name))
        exit()
        dataset = None

    return dataset
'''
from framework.datasets.XFGDataset_build import XFGSample,XFGBatch
def graph_collate_fn(samples: List[XFGSample]) -> XFGBatch:
    
    """
    Collate function for DataLoader.
    Args:
        samples (List[XFGSample]): List of XFGSamples
    Returns:
        XFGBatch: An object representing a batch of graphs and labels.
    """
    data = XFGBatch(XFGs=samples)
    return data.graphs, data.labels#XFGBatch(XFGs=samples)


'''
def graph_collate_fn(batch):
    # batch是一个列表，其中的元素是您的数据集__getitem__返回的数据
    # 例如：[(input_x1, label1), (input_x2, label2), ...]
    #print('print graph collate batch')
    #print(batch[0])
    #(Data(edge_index=[2, 172], x=[205, 101], y=[1]), tensor(1))
    # 分解输入和标签
    try:
        input_xs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        #print(labels,type(labels))
    except:
        print('Error in graph_collate_fn')
        for item in batch:
            print(item)
    # 我们不能简单地堆叠input_xs，因为edge_index的大小是不同的
    # 所以我们将其保留为一个列表
    data_list = []
    # try:
    for input_x in input_xs:
        edge_index, *remaining_data = input_x
        # print(edge_index,type(edge_index))
        # print(remaining_data, type(remaining_data))
        # 检查 edge_index 是否为 torch.Tensor，如果不是，则将其转换为 torch.Tensor
        #if not isinstance(edge_index, torch.Tensor):
        #    edge_index = torch.tensor(edge_index, dtype=torch.long)
        # 创建一个字典，其中 edge_index 是一个键，其余数据作为一个整体是另一个键
        data_dict = {"edge_index": edge_index[1], "my_data": remaining_data}

        # 将这个字典转换为 Data 对象
        data_object = Data(**data_dict)
        data_list.append(data_object)
        #data_list.append(Data(my_data=data, edge_index=edge_index))
    # except:
    #     print(input_xs)

    # labels是一个简单的tensor列表，我们可以直接堆叠它们
    labels = torch.stack(labels, dim=0)
    labels = labels.squeeze()

    return data_list, labels


def get_dataloader(config, split, dataset, batch_size, num_workers=1, **kw):
    if 'dataloader' in config and config['dataloader'] == 'geometric':
        if split == 'train':
            return DataLoader(dataset, collate_fn = graph_collate_fn, batch_size=batch_size, num_workers=num_workers, drop_last=True, pin_memory=False, sampler=kw['sampler'])
        return DataLoader(dataset, collate_fn=graph_collate_fn, batch_size=1, num_workers=1, pin_memory=True)
    
    #sequence dataset
    if split == 'train':
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, pin_memory=False, sampler=kw['sampler'])
    return DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True)
