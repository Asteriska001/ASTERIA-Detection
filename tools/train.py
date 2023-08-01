import torch 
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist

from framework.models import *
from framework.datasets import * 
#from framework.augmentations import get_train_augmentation, get_val_augmentation
from framework.model import get_model
from framework.dataset import get_dataset
from framework.losses import get_loss
from framework.schedulers import get_scheduler
from framework.optimizers import get_optimizer
from framework.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from val import evaluate

def main(cfg, gpu, save_dir):
    start = time.time()
    #best_mIoU = 0.0
    best_Acc = 0.0

    num_workers = mp.cpu_count()
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    
    #traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    #valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    #traintransform = ''
    #valtransform = ''

    #important
    #trainset = get_dataset(dataset_cfg , 'train')
    #valset = get_dataset(dataset_cfg , 'val')
    #trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train')
    #valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val')

    '''
    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform)
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform)
    '''
    

    model = get_model(model_cfg)
    print('test')
    print(model)
    #model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes)
    
    #pretrained support
    #model.init_pretrained(model_cfg['PRETRAINED'])
    model = model.to(device)

    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        model = DDP(model, device_ids=[gpu])
    else:
        sampler = RandomSampler(trainset)
    
    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=True, sampler=sampler)
    valloader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True)

    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE']
    # class_weights = trainset.class_weights.to(device)
    #loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    loss_fn = get_loss(loss_cfg['NAME'])
    print(loss_fn)
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, epochs * iters_per_epoch, sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    scaler = GradScaler(enabled=train_cfg['AMP'])
    writer = SummaryWriter(str(save_dir / 'logs'))

    for epoch in range(epochs):
        model.train()
        if train_cfg['DDP']: sampler.set_epoch(epoch)

        train_loss = 0.0
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        for iter, (input_x, lbl) in pbar:
            optimizer.zero_grad(set_to_none=True)

            input_x = input_x.to(device)
            lbl = lbl.to(device)
            
            with autocast(enabled=train_cfg['AMP']):
                logits = model(input_x)
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
            #miou = evaluate(model, valloader, device)[-1]
            #writer.add_scalar('val/mIoU', miou, epoch)
            #if acc > best_acc:
            '''
            if miou > best_mIoU:
                best_mIoU = miou
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth")
            print(f"Current mIoU: {miou} Best mIoU: {best_mIoU}")
            '''

    writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)
    '''
    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    print(tabulate(table, numalign='right'))
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    #fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    main(cfg, gpu, save_dir)
    cleanup_ddp()