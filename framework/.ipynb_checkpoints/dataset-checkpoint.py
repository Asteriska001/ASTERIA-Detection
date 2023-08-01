import torch
from torch import nn

from framework.datasets import *

#手动构建字典
DATASETS_DICT = {
    'REVEAL': REVEAL
}


# 自动构建字典
#DATASETS_DICT = {k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, nn.Module)}


def get_dataset(config, split):
    dataset_name = config['NAME']  # 这个应该是从配置文件中读取的模型名称
    dataset_param = config['PARAMS']  # 这个应该是从配置文件中读取的模型参数
    #待测试 不知道能不能直接把model——param的字典转为参数传递
    print(dataset_name)
    print(dataset_param)

    if dataset_name in DATASETS_DICT:
        dataset = DATASETS_DICT[dataset_name](split = split, root = config['ROOT'], **dataset_param)
    else:
        print("The dataset name {} does not exist".format(dataset_name))
        dataset = None

    return dataset
