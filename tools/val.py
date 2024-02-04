import torch
import argparse
import yaml
import math
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F

from framework.models import *
from framework.datasets import *
from framework.metrics import Metrics
from framework.utils.utils import setup_cudnn
from framework.dataset import get_dataset
from framework.model import get_model

def convert_output(pred):
    # 将一维预测转换为二维
    return torch.stack([1 - pred, pred], dim=1)

@torch.no_grad()
def evaluate(model, dataloader, device):
    print('Evaluating...')
    model.eval()
    metrics = Metrics(dataloader.dataset.n_classes, device)
    for input_x, labels in tqdm(dataloader):
        input_x = input_x.to(device)
        preds = model(input_x).to(device)
        #preds = convert_output(preds)
        labels = torch.tensor([labels])
        labels = labels.to(device)
        # print('preds value and shape: ',preds,preds.shape)
        # print('labels value and shape: ',labels, labels.shape)
        metrics.update(preds, labels)
    
    acc = metrics.compute_acc()
    print('acc: ',acc)
    f1 = metrics.compute_f1()
    print('f1: ',f1)
    rec = metrics.compute_rec()
    print('rec: ',rec)
    prec = metrics.compute_prec()
    print('prec: ',prec)
    roc_auc = metrics.compute_roc_auc()
    print('roc_auc: ',roc_auc)
    pr_auc = metrics.compute_pr_auc()
    print('pr_auc: ',pr_auc)

    return acc, f1, rec, prec, roc_auc, pr_auc

def main(cfg):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    #transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    valset = get_dataset(eval_cfg , 'val')
    #dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform)
    dataloader = DataLoader(valset, 1, num_workers=1, pin_memory=True)

    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): model_path = Path(cfg['SAVE_DIR']) / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth"
    print(f"Evaluating {model_path}...")

    model_cfg = cfg['MODEL']
    model = get_model(model_cfg)#待测试
    #model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], valset.n_classes)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    model = model.to(device)

    #if eval_cfg['MSF']['ENABLE']:
    #    acc, f1, rec, prec, roc_auc, pr_auc = evaluate_msf(model, dataloader, device, eval_cfg['MSF']['SCALES'], eval_cfg['MSF']['FLIP'])
    #else:
    acc, f1, rec, prec, roc_auc, pr_auc = evaluate(model, dataloader, device)

    table = {
        'Class': list(valset.CLASSES) + ['Mean'],
        'F1': f1 + [f1],
        'Acc': acc + [acc],
        'Rec': rec + [rec],
        'Prec': prec + [prec],
        'roc_auc': roc_auc + [roc_auc],
        'pr_auc': pr_auc +[pr_auc]
    }
    print(tabulate(table, headers='keys'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)