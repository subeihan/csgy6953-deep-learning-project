"""
train_isd.py

Reference:
[1] Ajinkya Tejankar1,Soroush Abbasi Koohpayegani, Vipin Pillai, Paolo Favaro, Hamed Pirsiavash
    ISD: Self-Supervised Learning by Iterative Similarity Distillation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import sampler
import torch.optim as optim
from tensorboardX import SummaryWriter

import os
import argparse
from tqdm import tqdm

from PIL import ImageFilter

from utils import *
from model import *
from isd import *

# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_option():

    # argument parser for running program from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=3407,
                        help='seed for torch')
    parser.add_argument('--root_path', type=str, default='./data',
                        help='data_path')
    parser.add_argument('--save_path', type=str, default='./model',
                        help='save_path')
    parser.add_argument('--save_every_e', type=int, default=10,
                        help='save model per save_every_e epoch')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size per gpu')
    parser.add_argument('--max_epoch', type=int, default=200,
                        help='maximum epoch number to train')
    parser.add_argument('--arc_opt', type=int, default=2,
                        help='2: num_planes=[64,128,256,512], num_blocks=[2,1,1,1];\
                        1: num_planes=[32,64,128,256], num_blocks=[2,2,2,2]')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--l2_reg', default=False, action='store_true')
    parser.add_argument('--no-l2_reg', dest='l2_reg', action='store_false')
    parser.add_argument('--adjust_lr', default=False, action='store_true')
    parser.add_argument('--no-adjust_lr', dest='adjust_lr', action='store_false')

    parser.add_argument('--checkpoint_path', default='./model', type=str,
                        help='where to save checkpoints.')
    parser.add_argument('--resume_path', default='', type=str,
                        help='path to latest checkpoint (default: none)')

    # isd hypers
    parser.add_argument('--K', type=int, default=12800,
                        help='size of memory bank')
    parser.add_argument('--m', type=float, default=0.999,
                        help='exponential momentum')
    parser.add_argument('--T', type=float, default=0.07,
                        help='temperature on output logits')

    parser.add_argument('--augmentation', type=str, default='weak/strong',
                        choices=['weak/strong', 'weak/weak', 'strong/weak', 'strong/strong'],
                        help='augmentation combo')


    args = parser.parse_args()

    return args

# Create train loader
def get_train_loader(args):
    image_size = 32
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    normalize = transforms.Normalize(mean=mean, std=std)

    aug_strong = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2))
        ]), p=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    aug_weak = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    if args.augmentation == 'weak/strong':
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.root_path,
            train=True,
            download=True,
            transform=TwoCropsTransform(k_t=aug_weak, q_t=aug_strong)
        )
    elif args.augmentation == 'weak/weak':
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.root_path,
            train=True,
            download=True,
            transform=TwoCropsTransform(k_t=aug_weak, q_t=aug_weak)
        )
    elif args.augmentation == 'strong/weak':
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.root_path,
            train=True,
            download=True,
            transform=TwoCropsTransform(k_t=aug_strong, q_t=aug_weak)
        )
    else: # strong/strong
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.root_path,
            train=True,
            download=True,
            transform=TwoCropsTransform(k_t=aug_strong, q_t=aug_strong)
        )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True)

    return train_loader


def main():
    args = parse_option()
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # prepare train_loader
    train_loader = get_train_loader(args)

    ## initialize the model
    model = MyResNet()
    if args.arc_opt == 1:
        model = MyResNet(block=BasicBlock,
                         num_planes=[32, 64, 128, 256],
                         num_blocks=[2, 2, 2, 2])
    else:
        model = MyResNet(block=BasicBlock,
                         num_planes=[64, 128, 256, 512],
                         num_blocks=[2, 1, 1, 1])

    # remove linear layer
    model.linear = nn.Sequential()
    model = model.to(DEVICE)

    isd = ISD(model, K=args.K, m=args.m, T=args.T).to(DEVICE)

    print(isd)

    criterion = KLD().to(DEVICE)

    params = [p for p in isd.parameters() if p.requires_grad]
    weight_decay = 1e-6 if args.l2_reg else 0
    optimizer = optim.Adam(isd.parameters(),
                           lr=args.learning_rate,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=weight_decay)

    args.start_epoch = 1

    if args.resume_path:
        print('==> resume from checkpoint: {}'.format(args.resume_path))
        ckpt = torch.load(args.resume_path)
        print('==> resume from epoch: {}'.format(ckpt['epoch']))
        isd.load_state_dict(ckpt['state_dict'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        args.start_epoch = ckpt['epoch'] + 1


    # train isd
    train(args, train_loader, isd, criterion, optimizer)

def train(args, train_loader, isd, criterion, optimizer):
    # visualization tool
    writer = SummaryWriter(args.save_path + '/log')

    max_iterations = args.max_epoch * len(train_loader)
    iteration = 0

    # record the best(lowest) loss
    cur_best_loss = 10000

    for ep in tqdm(range(args.start_epoch, args.max_epoch + 1)):

        # train student
        isd.train()
        total_train_loss = 0.0

        for idx, data in tqdm(enumerate(train_loader)):
            (im_q, im_k), _ = data
            im_q, im_k = im_q.to(DEVICE), im_k.to(DEVICE)

            # ===================forward=====================
            _, sim_q, sim_k = isd(im_q=im_q, im_k=im_k)
            loss = criterion(inputs=sim_q, targets=sim_k)

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===================meters=====================
            total_train_loss += loss.item()

            ## adjust learning rate (cosine scheduler) if asjust_lr set to be True
            lr_ = args.learning_rate
            if args.adjust_lr:
                lr_ = args.learning_rate * (1.0 - iteration / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            writer.add_scalar('info/lr', lr_, iteration)
            writer.add_scalar('info/isd_train_loss', loss, iteration)

            iteration += 1

        per_train_loss = total_train_loss / len(train_loader)

        if per_train_loss <= cur_best_loss:
            cur_best_loss = per_train_loss
            print('==> Saving...')
            state = {
                'opt': args,
                'state_dict': isd.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': ep,
            }

            save_file = os.path.join(args.save_path, 'best_isd.pth')
            torch.save(state, save_file)

            # help release GPU memory
            del state
            torch.cuda.empty_cache()


        # saving the model
        if ep % args.save_every_e == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'state_dict': isd.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': ep,
            }

            save_file = os.path.join(args.checkpoint_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=ep))
            torch.save(state, save_file)

            # help release GPU memory
            del state
            torch.cuda.empty_cache()

        print(f'epoch: {ep + 1:03}')
        print(f'\ttrain Loss: {per_train_loss:.3f}')

    print('training finished')
    writer.close()


if __name__ == '__main__':
    main()
