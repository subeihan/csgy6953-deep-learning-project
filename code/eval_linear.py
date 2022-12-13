import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

import model
from tools import *
from utils import *

parser = argparse.ArgumentParser(description='Unsupervised distillation')
parser.add_argument('--data_path', default='./data', type=str, metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-a', '--arch', default='MyResNet',
                    help='model architecture: | [MyResNet, ResNet18] (default: MyResNet)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=90, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume_path', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=3407, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save_path', default='./output/eval_linear', type=str,
                    help='output directory')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--weights_path', dest='weights_path', type=str, required=True,
                    help='pre-trained model weights')
parser.add_argument('--lr_schedule', type=str, default='15,30,40',
                    help='lr drop schedule')

best_acc1 = 0

# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    global logger

    args = parser.parse_args()
    makedirs(args.save_path)
    logger = get_logger(logpath=os.path.join(args.save_path, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    main_worker(args)

import pdb


def load_weights(backbone, wts_path):
    wts = torch.load(wts_path)

    ckpt = wts['state_dict']
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    state_dict = {}

    for m_key, m_val in backbone.state_dict().items():
        if m_key in ckpt:
            state_dict[m_key] = ckpt[m_key]
        else:
            state_dict[m_key] = m_val
            print('not copied => ' + m_key)

    backbone.load_state_dict(state_dict)
    print(backbone)


def get_backbone(arch, wts_path):
    backbone = model.__dict__[arch]()
    backbone.linear = nn.Sequential()
    load_weights(backbone, wts_path)

    for p in backbone.parameters():
        p.requires_grad = False

    return backbone


def get_dataloaders(args):
    # seed torch, np and random
    torch.cuda.empty_cache()
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        np.random.RandomState(worker_seed)
        random.seed(worker_seed)


    # perform data transformation on train_set and val_set
    image_size = 32
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # divide the overall train_set into train and val parts, with 80%:20% ratio
    num_train = 40000

    # prepare train dataset and dataloader
    train_dataset = datasets.CIFAR10(
        root=args.data_path,
        train=True,
        download=True,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler.SubsetRandomSampler(range(num_train)),
        worker_init_fn=seed_worker,
        pin_memory=True,
    )

    # prepare validation dataset and dataloader
    val_dataset = datasets.CIFAR10(
        root=args.data_path,
        train=True,
        download=True,
        transform=test_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=args.workers,
        sampler=sampler.SubsetRandomSampler(range(num_train, 50000)),
        pin_memory=True,
    )

    # prepare test dataset and dataloader
    test_dataset = datasets.CIFAR10(
        root=args.data_path,
        train=False,
        download=True,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader, test_loader

def main_worker(args):
    global best_acc

    # get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args)

    # get pretained backbone
    backbone = get_backbone(args.arch, args.weights_path)
    backbone = nn.DataParallel(backbone).to(DEVICE)
    backbone.eval()

    # linear head module
    linear = nn.Sequential(
        #nn.BatchNorm1d(512),
        nn.Linear(512, 10)
    )
    linear = linear.to(DEVICE)

    optimizer = optim.Adam(linear.parameters(),
                           lr=args.lr,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=args.weight_decay)


    schedule = [int(x) for x in args.lr_schedule.split(',')]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=schedule
    )

    # optionally resume from a checkpoint
    if args.resume_path:
        if os.path.isfile(args.resum_path):
            logger.info("=> loading checkpoint '{}'".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            args.start_epoch = checkpoint['epoch']
            linear.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resum_path, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume_path))

    cudnn.benchmark = True

    # evaluate on test set if run in evaluation mode
    if args.evaluate:
        validate(test_loader, backbone, linear, args)
        return

    # train
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, backbone, linear, optimizer, epoch, args)

        # evaluate on validation set
        val_acc = validate(val_loader, backbone, linear, args)

        # modify lr
        lr_scheduler.step()

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': linear.state_dict(),
            'best_acc1': best_acc,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, is_best, args.save_path)

    # test
    validate(test_loader, backbone, linear, args)



def train(train_loader, backbone, linear, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    backbone.eval()
    linear.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(DEVICE)
        target = target.to(DEVICE)

        # compute output
        with torch.no_grad():
            output, _ = backbone(images)
        output = linear(output)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do ADAM step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(progress.display(i))


def validate(val_loader, backbone, linear, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    backbone.eval()
    linear.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(DEVICE)
            target = target.to(DEVICE)

            # compute output
            output = backbone(images)
            output = linear(output)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(progress.display(i))

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

if __name__ == '__main__':
    main()
