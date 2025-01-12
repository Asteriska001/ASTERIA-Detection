import torch 
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import Data


from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist

from framework.models import *
from framework.datasets import * 
from framework.model import get_model
from framework.dataset import get_dataset, get_dataloader
from framework.losses import get_loss
from framework.schedulers import get_scheduler
from framework.optimizers import get_optimizer
from framework.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from val import evaluate

#ordered load yaml files
from collections import OrderedDict

from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()


def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

def main(cfg, gpu, save_dir):
    start = time.time()
    best_Acc = 0.0

    num_workers = mp.cpu_count()
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    
    #important
    trainset = get_dataset(cfg , 'train')
    valset = get_dataset(cfg , 'val')
    #todo fine tuning
    #program modelling
        #features extraction
        #encoder -> encodings
    
    #trainset = get_representation(cfg)
    #valset = get_representation(cfg)

    model = get_model(model_cfg)
    print(model)
    
    #pretrained support
    #model.init_pretrained(model_cfg['PRETRAINED'])
    '''
    if model_cfg['PRETRAINED'] != None:
        model = get_pretrained_model(model_cfg['PRETRAINED'])
    '''
    model = model.to(device)

    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        model = DDP(model, device_ids=[gpu])
    else:
        sampler = RandomSampler(trainset)
    
    #trainloader = DataLoader(trainset, collate_fn = custom_collate_fn, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=False, sampler=sampler)
    #valloader = DataLoader(valset, collate_fn=custom_collate_fn, batch_size=1, num_workers=1, pin_memory=True)
    trainloader = get_dataloader(dataset_cfg, 'train', trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, sampler=sampler)
    valloader = get_dataloader(dataset_cfg, 'val', valset, batch_size=1, num_workers=1)

    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE']
    # class_weights = trainset.class_weights.to(device)
    loss_fn = get_loss(loss_cfg['NAME'])
    print('loss func: '+ str(loss_fn))
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, epochs * iters_per_epoch, sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    scaler = GradScaler(enabled=train_cfg['AMP'])
    writer = SummaryWriter(str(save_dir / 'logs'))

    for epoch in range(epochs):
        model.train()
        if train_cfg['DDP']: sampler.set_epoch(epoch)

        train_loss = 0.0
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")
        for iter, data in pbar:
        #for iter, (input_x, lbl) in pbar:
            #print('pbar data')
            #print(data)
            #input_x = data.graphs
            #lbl = data.labels
            #print(data)
            (input_x,lbl) = data
            #print('input_x data')
            #print(input_x)
            #print("train max input:", torch.max(input_x[0]))
            
            optimizer.zero_grad(set_to_none=True)

            #input_x = tuple(torch.tensor(item).to(device) if isinstance(item, list) else item.to(device) for item in input_x)
            #tuple(tensor.to(device) for tensor in input_x)
            try:
                input_x = input_x.to(device)
                #input_x = [torch.tensor(x).to(device) for x in input_x]#input_x.to(device)
                # input_x_transformed = []
                # for x in input_x:
                #     if isinstance(x, torch.Tensor):
                #         # 对于 PyTorch 张量，直接转移到设备
                #         x_transformed = x.to(device)
                #     elif isinstance(x, Data):
                #         # 对于 PyTorch Geometric 的 Data 对象，也直接转移到设备
                #         x_transformed = x.to(device)
                #     else:
                #         # 对于其他类型，根据需要进行处理
                #         # 例如，如果是 NumPy 数组，先转换为张量，然后转移到设备
                #         x_transformed = torch.tensor(x).to(device)
                #     input_x_transformed.append(x_transformed)
                # input_x = torch.tensor(input_x_transformed).to(device)
            except:
                pass
                # print('Error in the inputx to device:')
                # for x in input_x:
                #     print(x,type(x))
            lbl = lbl.to(device)
            #print('input shape: '+str(input_x))
            with autocast(enabled=train_cfg['AMP']):
                # print(model.device)
                # print(input_x.device)
                # print(lbl.device)
                logits = model(input_x)
                #data, edge_index = input_x
                #edge_index_list = [data.edge_index for data in input_x]
                #my_data_list = [data.my_data for data in input_x]
                #logits = model(my_data_list, edge_index_list)
                #print(logits.shape)
                #print(lbl.shape)
                # print('logits.shape: ',logits.shape)
                # print('lbl.shape: ', lbl.shape)
                loss = loss_fn(logits, lbl)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            train_loss += loss.item()

            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
        
        train_loss /= iter+1
        writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()
        #eval_interval 
        if (epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 or (epoch+1) == epochs:
            print('eval_interval:')
            acc, f1, rec, prec, roc_auc, pr_auc = evaluate(model, valloader, device)
            writer.add_scalar('val/acc', acc, epoch)
            if acc > best_Acc:
                best_Acc = acc
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth")
            print(f"Current Accuracy: {acc} Best Accuracy: {best_Acc}")

    writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)
    table = [
        ['Best Acc', f"{best_Acc:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    print(tabulate(table, numalign='right'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        ordered_dict = ordered_load(f, yaml.SafeLoader)
        cfg = ordered_dict
    #with open(args.cfg) as f:
    #    cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(123456)#fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    main(cfg, gpu, save_dir)
    cleanup_ddp()
