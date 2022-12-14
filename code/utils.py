"""
utils.py
utility functions for training and testing

Reference:
[1] Ajinkya Tejankar1,Soroush Abbasi Koohpayegani, Vipin Pillai, Paolo Favaro, Hamed Pirsiavash
    ISD: Self-Supervised Learning by Iterative Similarity Distillation
"""

from __future__ import print_function
import math

import torch
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(outputs, labels):
    total = 0
    correct = 0
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    return correct/total

class TwoCropsTransform:
    def __init__(self, k_t, q_t):
        self.q_t = q_t
        self.k_t = k_t
        print('======= Query transform =======')
        print(self.q_t)
        print('===============================')
        print('======== Key transform ========')
        print(self.k_t)
        print('===============================')

    def __call__(self, x):
        q = self.q_t(x)
        k = self.k_t(x)
        return [q, k]

if __name__ == '__main__':
    meter = AverageMeter()