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

def get_max_cuda_memory() -> int:
    """Returns the maximum GPU memory occupied by tensors in megabytes (MB) for
    a given device. By default, this returns the peak allocated memory since
    the beginning of this program.

    Args:
        device (torch.device, optional): selected device. Returns
            statistic for the current device, given by
            :func:`~torch.cuda.current_device`, if ``device`` is None.
            Defaults to None.

    Returns:
        int: The maximum GPU memory occupied by tensors in megabytes
        for a given device.
    """
    mem = torch.cuda.max_memory_allocated(device='cuda')
    mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                          dtype=torch.int,
                          device='cuda')
    torch.cuda.reset_peak_memory_stats()
    return int(mem_mb.item())

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

    #for epoch in range(epochs):
    
    model.train()
    if train_cfg['DDP']: sampler.set_epoch(epoch)

    train_loss = 0.0
    nb_eval_steps = 0
    warmup_steps = 5
    total_steps = 3000
    exit_flag = False

    pure_inf_time = 0
    
    while True:
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc='')#f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")
        for iter, data in pbar:
            (input_x,lbl) = data
            optimizer.zero_grad(set_to_none=True)

            try:
                input_x = input_x.to(device)
            except:
                print('Error in the inputx to device:')
                for x in input_x:
                    print(x,type(x))
            lbl = lbl.to(device)
            
            nb_eval_steps += 1
            
            with autocast(enabled=train_cfg['AMP']):
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                logits = model(input_x)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                loss = loss_fn(logits, lbl)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            train_loss += loss.item()

            #pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
            
            if nb_eval_steps >= warmup_steps:
                pure_inf_time += elapsed
            if nb_eval_steps==total_steps+warmup_steps:
                fps = total_steps/ pure_inf_time
                print(
                f'Overall fps: {fps:.1f} data / s')
                print('Train infer',pure_inf_time/total_steps)
                
                cuda_memory = get_max_cuda_memory()
                print(f'Cuda memory: {cuda_memory} MB')
                print(f'Cuda memory: {cuda_memory/1024}G')
                exit_flag= True
                break
        if exit_flag: 
            break

        train_loss /= iter+1
        #writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()
        # #eval_interval 
        # if (epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 or (epoch+1) == epochs:
        #     print('eval_interval:')
        #     acc, f1, rec, prec, roc_auc, pr_auc = evaluate(model, valloader, device)
        #     writer.add_scalar('val/acc', acc, epoch)
        #     if acc > best_Acc:
        #         best_Acc = acc
        #         torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth")
        #     print(f"Current Accuracy: {acc} Best Accuracy: {best_Acc}")

    # writer.close()
    # pbar.close()
    # end = time.gmtime(time.time() - start)
    # table = [
    #     ['Best Acc', f"{best_Acc:.2f}"],
    #     ['Total Training Time', time.strftime("%H:%M:%S", end)]
    # ]
    # print(tabulate(table, numalign='right'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        ordered_dict = ordered_load(f, yaml.SafeLoader)
        cfg = ordered_dict

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    main(cfg, gpu, save_dir)
    cleanup_ddp()
