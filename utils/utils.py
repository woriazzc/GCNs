import os
import sys
from datetime import date, datetime
import random
import pyro
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_np(x):
    return x.detach().data.cpu().numpy()


def seed_all(seed:int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def avg_dict(eval_dicts, final_dict=None):
    if final_dict is None:
        final_dict = {}
    flg_dict = eval_dicts[0]
    for k in flg_dict:
        if isinstance(flg_dict[k], dict):
            final_dict[k] = avg_dict([eval_dict[k] for eval_dict in eval_dicts])
        else:
            final_dict[k] = 0
            for eval_dict in eval_dicts:
                final_dict[k] += eval_dict[k]
            final_dict[k] /= len(eval_dicts)
    return final_dict


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

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

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


class Logger:
    def __init__(self, root_dir, dataset, model, suffix, no_log):
        self.log_dir = os.path.join(root_dir, dataset)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, model + ('_' if suffix != '' else '') + suffix + '.log')
        self.no_log = no_log

    def log(self, content='', pre=True, end='\n'):
        string = str(content)
        if len(string) == 0:
            pre = False
        if pre:
            today = date.today()
            today_date = today.strftime("%m/%d/%Y")
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            string = today_date + "," + current_time + ": " + string
        string = string + end

        if not self.no_log:
            with open(self.log_path, 'a') as logf:
                logf.write(string)

        sys.stdout.write(string)
        sys.stdout.flush()
    
    def log_args(self, args):
        self.log('-' * 40 + 'ARGUMENTS' + '-' * 40, pre=False)
        for arg in vars(args):
            self.log('{:40} {}'.format(arg, getattr(args, arg)), pre=False)
        self.log('-' * 40 + 'ARGUMENTS' + '-' * 40, pre=False)


# Some graph things

def convert_pairs_to_graph(pairs, num_users, num_items):
    user_dim = torch.LongTensor(pairs[:, 0].cpu())
    item_dim = torch.LongTensor(pairs[:, 1].cpu())

    first_sub = torch.stack([user_dim, item_dim + num_users])
    second_sub = torch.stack([item_dim + num_users, user_dim])
    index = torch.cat([first_sub, second_sub], dim=1)
    data = torch.ones(index.size(-1)).int()
    Graph = torch.sparse.IntTensor(index, data,
                                    torch.Size([num_users + num_items, num_users + num_items]))
    dense = Graph.to_dense()
    D = torch.sum(dense, dim=1).float()
    D[D == 0.] = 1.
    D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
    dense = dense / D_sqrt
    dense = dense / D_sqrt.t()
    index = dense.nonzero(as_tuple =False)
    data = dense[dense >= 1e-9]
    assert len(index) == len(data)
    Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size(
        [num_users + num_items, num_users + num_items]))
    Graph = Graph.coalesce().cuda()
    return Graph
    

def cal_spectral_feature(mat, k, largest=True, niter=10):
    """
    Compute spectual values and vectors
    mat: torch.Tensor, sparse or dense
    k: the number of required features
    largest: (default: True) True for k-largest and False for k-smallest eigenvalues
    niter: maximum number of iterations
    """
    value, vector = torch.lobpcg(mat, k=k, largest=largest, niter=niter)
    return value, vector


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape), dtype=torch.float32)
