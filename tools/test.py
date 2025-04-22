import os
import torch
import torch.nn as nn
import logging
import time
import numpy as np
from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
sys.path.append(os.getcwd())
from lib.models.builder import build_model
from lib.dataset.builder import build_dataloader
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger
from lib.utils.misc import accuracy, AverageMeter, CheckpointManager
from lib.utils.model_ema import ModelEMA
from lib.utils.measure import get_params, get_flops

torch.backends.cudnn.benchmark = True

'''init logger'''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    args, args_text = parse_args()
    args.exp_dir = f'experiments/{args.experiment}'

    '''distributed'''
    init_dist(args)
    init_logger(args)

    '''build dataloader'''
    _, val_dataset, _, val_loader = \
        build_dataloader(args)

    '''build model'''
    loss_fn = nn.CrossEntropyLoss().cuda()
    val_loss_fn = loss_fn

    model = build_model(args, args.model)
    logger.info(model)
    logger.info(
        f'Model {args.model} created, params: {get_params(model) / 1e6:.3f} M, '
        f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')

    model.cuda()
    model = DDP(model,
                device_ids=[args.local_rank],
                find_unused_parameters=False)

    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None

    '''resume'''
    ckpt_manager = CheckpointManager(model,
                                     ema_model=model_ema,
                                     save_dir=args.exp_dir,
                                     rank=args.rank)

    if args.resume:
        epoch = ckpt_manager.load(args.resume)
        logger.info(
            f'Resume ckpt {args.resume} done, '
            f'epoch {epoch}'
        )
    else:
        epoch = 0

    # validate
    test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)
    if model_ema is not None:
        test_metrics = validate(args,
                                epoch,
                                model_ema.module,
                                val_loader,
                                loss_fn,
                                log_suffix='(EMA)')
    logger.info(test_metrics)

    # classroom model metrics
    if args.kd != '':
        # build teacher model
        if(len(args.teacher_model)):
            if(len(args.teacher_ckpt)):
                teacher_models = nn.ModuleList([build_model(args, args.teacher_model[i], args.teacher_pretrained, args.teacher_ckpt[i]) for i in range(len(args.teacher_model))])
            else:
                teacher_models = nn.ModuleList([build_model(args, args.teacher_model[i], args.teacher_pretrained) for i in range(len(args.teacher_model))])
        else:
            teacher_models = nn.ModuleList()
        if(len(args.peer_ckpt)):
            peer_models =  nn.ModuleList([build_model(args, args.peer_model[i], args.peer_pretrained, args.peer_ckpt[i]) for i in range(len(args.peer_model))])
        else:
            peer_models =  nn.ModuleList([build_model(args, args.peer_model[i], args.peer_pretrained) for i in range(len(args.peer_model))])
        # for i, teacher_model in enumerate(teacher_models):
        #     logger.info(
        #         f'Teacher model {i} of {args.teacher_model} created, params: {get_params(teacher_model) / 1e6:.3f} M, '
        #         f'FLOPs: {get_flops(teacher_model, input_shape=args.input_shape) / 1e9:.3f} G')
        #     teacher_model.cuda()
        #     test_metrics = validate(args, 0, teacher_model, val_loader, val_loss_fn, log_suffix=' (teacher){i}')
        #     logger.info(f'Top-1 accuracy of teacher model {i}: {test_metrics["top1"]:.2f}')
        
        # for i,peer_model in enumerate(peer_models):
        #     logger.info(
        #         f'Peer model {i} of {args.peer_model} created, params: {get_params(peer_model) / 1e6:.3f} M, '
        #         f'FLOPs: {get_flops(peer_model, input_shape=args.input_shape) / 1e9:.3f} G')
        #     peer_model.cuda()
        #     test_metrics = validate(args, 0, peer_model, val_loader, val_loss_fn, log_suffix=' (peer){i}')
        #     logger.info(f'Top-1 accuracy of peer model {i}: {test_metrics["top1"]:.2f}')
        ensemble = nn.ModuleList(list(teacher_models) + list(peer_models))
        test_metrics = validate_ensemble(args, 0, ensemble, val_loader, val_loss_fn)
        logger.info(f'Top-1 accuracy of ensemble: {test_metrics["top1"]:.2f}')

class AverageMeter:
    def __init__(self, dist=False):
        self.reset()
        self.dist = dist

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(args, epoch, model, loader, loss_fn, log_suffix=''):
    loss_m = AverageMeter(dist=True)
    top1_m = AverageMeter(dist=True)
    top5_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()

    model.eval()
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for batch_idx, (input, target) in enumerate(loader):
        with torch.no_grad():
            output = model(input)
            loss = loss_fn(output, target)

        top1, top5 = accuracy(output, target, topk=(1, 5))
        loss_m.update(loss.item(), n=input.size(0))
        top1_m.update(top1.item(), n=input.size(0))
        top5_m.update(top5.item(), n=input.size(0))

        _, predicted = torch.max(output, 1)
        correct = (predicted == target).squeeze()
        
        for i in range(len(target)):
            label = target[i].item()
            class_correct[label] += correct[i].item()
            class_total[label] += 1

        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Test{}: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'Top-1: {top1.val:.3f}% ({top1.avg:.3f}%) '
                        'Top-5: {top5.val:.3f}% ({top5.avg:.3f}%) '
                        'Time: {batch_time.val:.2f}s'.format(
                            log_suffix,
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m))
        start_time = time.time()

    per_class_accuracy = {class_idx: 100.0 * class_correct[class_idx] / class_total[class_idx]
                          for class_idx in class_correct.keys()}
    
    for class_idx in sorted(per_class_accuracy.keys()):
        logger.info('Class {} Accuracy: {:.3f}%'.format(class_idx, per_class_accuracy[class_idx]))
    
    return {
        'test_loss': loss_m.avg, 
        'top1': top1_m.avg, 
        'top5': top5_m.avg,
        'per_class_accuracy': per_class_accuracy
    }

def validate_ensemble(args, epoch, models, loader, loss_fn, log_suffix=''):
    loss_m = AverageMeter(dist=True)
    top1_m = AverageMeter(dist=True)
    top5_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()

    for model in models:
        model.eval()
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for batch_idx, (input, target) in enumerate(loader):
        with torch.no_grad():
            outputs = []
            for model in models:
                model.eval()
                model.cuda()
                output_m = model(input)
                outputs.append(output_m)
            output = torch.stack(outputs).max(dim=0)[0]
            loss = loss_fn(output, target)

        top1, top5 = accuracy(output, target, topk=(1, 5))
        loss_m.update(loss.item(), n=input.size(0))
        top1_m.update(top1.item(), n=input.size(0))
        top5_m.update(top5.item(), n=input.size(0))

        _, predicted = torch.max(output, 1)
        correct = (predicted == target).squeeze()
        
        for i in range(len(target)):
            label = target[i].item()
            class_correct[label] += correct[i].item()
            class_total[label] += 1

        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Test{}: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'Top-1: {top1.val:.3f}% ({top1.avg:.3f}%) '
                        'Top-5: {top5.val:.3f}% ({top5.avg:.3f}%) '
                        'Time: {batch_time.val:.2f}s'.format(
                            log_suffix,
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m))
        start_time = time.time()

    per_class_accuracy = {class_idx: 100.0 * class_correct[class_idx] / class_total[class_idx]
                          for class_idx in class_correct.keys()}
    
    for class_idx in sorted(per_class_accuracy.keys()):
        logger.info('Class {} Accuracy: {:.3f}%'.format(class_idx, per_class_accuracy[class_idx]))
    
    return {
        'test_loss': loss_m.avg, 
        'top1': top1_m.avg, 
        'top5': top5_m.avg,
        'per_class_accuracy': per_class_accuracy
    }


if __name__ == '__main__':
    main()
