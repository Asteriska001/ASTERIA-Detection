from tabulate import tabulate
from framework import models
from framework import datasets
#from framework.models import backbones, heads

#from framework import representations


def show_models():
    model_names = models.__all__
    numbers = list(range(1, len(model_names)+1))
    print(tabulate({'No.': numbers, 'Model Names': model_names}, headers='keys'))

def show_representations():
    print(type(representations))
    representation_names = representations.__all__
    numbers = list(range(1, len(representation_names)+1))
    print(tabulate({'No.': numbers, 'Representation Names': representation_names}, headers='keys'))

'''
def show_backbones():
    backbone_names = backbones.__all__
    variants = []
    for name in backbone_names:
        try:
            variants.append(list(eval(f"backbones.{name.lower()}_settings").keys()))
        except:
            variants.append('-')
    print(tabulate({'Backbone Names': backbone_names, 'Variants': variants}, headers='keys'))


def show_heads():
    head_names = heads.__all__
    numbers = list(range(1, len(head_names)+1))
    print(tabulate({'No.': numbers, 'Heads': head_names}, headers='keys'))
'''

def show_datasets():
    dataset_names = datasets.__all__
    numbers = list(range(1, len(dataset_names)+1))
    print(tabulate({'No.': numbers, 'Datasets': dataset_names}, headers='keys'))