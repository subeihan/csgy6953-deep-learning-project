import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class KLD(nn.Module):
    def forward(self, inputs, targets):
        inputs = F.log_softmax(inputs, dim=1)
        targets = F.softmax(targets, dim=1)
        return F.kl_div(inputs, targets, reduction='batchmean')


def get_mlp(inp_dim, hidden_dim, out_dim):
    mlp = nn.Sequential(
        nn.Linear(inp_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )
    return mlp

def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


class ISD(nn.Module):
    def __init__(self, arch, K=65536, m=0.999, T=0.07):
        super(ISD, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create encoders and projection layers        
        self.encoder_q = copy.deepcopy(arch)
        self.encoder_k = copy.deepcopy(arch)
        feat_dim = self.encoder_q.feat_size
        out_dim = feat_dim
        
        ##### prediction layer ####
        # 1. have a prediction layer for q with BN
        self.predict_q = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=True),
        )
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # setup queue
        self.register_buffer('queue', torch.randn(self.K, out_dim))
        # normalize the queue
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # setup the queue pointer
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
    @torch.no_grad()
    def data_parallel(self):
        self.encoder_q = torch.nn.DataParallel(self.encoder_q)
        self.encoder_k = torch.nn.DataParallel(self.encoder_k)
        self.predict_q = torch.nn.DataParallel(self.predict_q)
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0 

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
        
    def forward(self, im_q, im_k):
        res, feat_q = self.encoder_q(im_q)
        q = self.predict_q(feat_q)
        q = nn.functional.normalize(q, dim=1)
        # compute key features
        with torch.no_grad():
            # update the key encoder
            self._momentum_update_key_encoder()

            # shuffle keys
            shuffle_ids, reverse_ids = get_shuffle_ids(im_k.shape[0])
            im_k = im_k[shuffle_ids]

            # forward through the key encoder
            _, k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = k[reverse_ids]
            
        # calculate similarities
        queue = self.queue.clone().detach()
        sim_q = torch.mm(q, queue.t())
        sim_k = torch.mm(k, queue.t())

        # scale the similarities with temperature
        sim_q /= self.T
        sim_k /= self.T

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return res, sim_q, sim_k