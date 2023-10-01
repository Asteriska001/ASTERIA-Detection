import torch
from torch import nn

from framework.datasets import *
from framework.datasets.XFGDataset_build import DWK_Dataset
from framework.preprocess import get_preprocess

#手动构建字典
DATASETS_DICT = {
    'REVEAL': REVEAL,
    'Devign_Partial': Devign_Partial,
    'CodeXGLUE': CodeXGLUE,
    'DWK_Dataset': DWK_Dataset,
    'IVDDataset': IVDetectDataset
}


# 自动构建字典
#DATASETS_DICT = {k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, nn.Module)}


def get_dataset(config, split):

    dataset_cfg = config['DATASET']
    train_cfg, eval_cfg = config['TRAIN'], config['EVAL']

    dataset_name = dataset_cfg['NAME']  # 这个应该是从配置文件中读取的模型名称
    dataset_param = dataset_cfg['PARAMS']  # 这个应该是从配置文件中读取的模型参数
    #待测试 不知道能不能直接把model——param的字典转为参数传递
    print(dataset_name)
    print(dataset_param)
    print(DATASETS_DICT)
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
        dataset = None

    return dataset
