import torch
from torch import nn

from framework.models import *

#手动构建字典
MODEL_DICT = {
    'GNNReGVD': GNNReGVD,
    'Devign': Devign,
}


# 自动构建字典
#MODEL_DICT = {k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, nn.Module)}


def get_model(config):
    model_name = config['NAME']  # 这个应该是从配置文件中读取的模型名称
    model_param = config['PARAMS']  # 这个应该是从配置文件中读取的模型参数
    #待测试 不知道能不能直接把model——param的字典转为参数传递
    if model_name in MODEL_DICT:
        model = MODEL_DICT[model_name](**model_param)
    else:
        print("The model name {} does not exist".format(model_name))
        model = None

    return model
