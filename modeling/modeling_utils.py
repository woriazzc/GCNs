import time
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_sparse import sum, mul, fill_diag, remove_diag
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GeneralPropagation(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int, alpha: float, dropout: float = 0.,
                 lambda2: float = 4.0,
                 beta: float = 0.5,
                 ogb: bool = True,
                 incnorm_para: bool = True,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 add_self_loops_l1: bool = True,
                 normalize: bool = True,
                 node_num: int = None,
                 num_classes: int = None,
                 **kwargs):

        super(GeneralPropagation, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        self.lambda2 = lambda2
        self.beta = beta
        self.ogb = ogb
        self.incnorm_para = incnorm_para
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.add_self_loops_l1 = add_self_loops_l1

        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_inc = None

        self.node_num = node_num
        self.num_classes = num_classes
        self.max_value = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_inc = None

    def get_incident_matrix(self, edge_index: Adj):
        size = edge_index.sizes()[1]
        row_index = edge_index.storage.row()
        col_index = edge_index.storage.col()
        mask = row_index >= col_index
        row_index = row_index[mask]
        col_index = col_index[mask]
        edge_num = row_index.numel()
        row = torch.cat([torch.arange(edge_num), torch.arange(edge_num)]).cuda()
        col = torch.cat([row_index, col_index])
        value = torch.cat([torch.ones(edge_num), -1 * torch.ones(edge_num)]).cuda()
        inc = SparseTensor(row=row, rowptr=None, col=col, value=value,
                           sparse_sizes=(edge_num, size))
        return inc

    def inc_norm(self, inc, edge_index, add_self_loops, normalize_para=-0.5):
        if add_self_loops:
            edge_index = fill_diag(edge_index, 1.0)
        else:
            edge_index = remove_diag(edge_index)
        deg = sum(edge_index, dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        inc = mul(inc, deg_inv_sqrt.view(1, -1))  ## col-wise
        return inc

    def check_inc(self, edge_index, inc, normalize=False):
        # return None  ## not checking it
        nnz = edge_index.nnz()
        if normalize:
            deg = torch.eye(edge_index.sizes()[0])  # .cuda()
        else:
            deg = sum(edge_index, dim=1).cpu()
            deg = torch.diag(deg)
        inc = inc.cpu()
        lap = (inc.t() @ inc).to_dense()
        adj = edge_index.cpu().to_dense()

        lap2 = deg - adj
        diff = torch.sum(torch.abs(lap2 - lap)) / nnz
        # import ipdb; ipdb.set_trace()
        assert diff < 0.000001, f'error: {diff} need to make sure L=B^TB'

    def forward(self, x: Tensor, edge_index: Adj, x_idx: Tensor = None,
                edge_weight: OptTensor = None, niter=None,
                data=None) -> Tensor:
        """"""
        start_time = time.time()
        edge_index2 = edge_index
        if self.normalize:
            if isinstance(edge_index, Tensor):
                raise ValueError('Only support SparseTensor now')
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                ## first cache incident_matrix (before normalizing edge_index)
                cache = self._cached_inc
                if cache is None:
                    incident_matrix = self.get_incident_matrix(edge_index=edge_index)
                    if not self.ogb:
                        self.check_inc(edge_index=edge_index, inc=incident_matrix, normalize=False)
                    incident_matrix = self.inc_norm(inc=incident_matrix, edge_index=edge_index,
                                                    add_self_loops=self.add_self_loops_l1,
                                                    normalize_para=self.incnorm_para)
                    if not self.ogb:
                        edge_index_C = gcn_norm(  # yapf: disable
                            edge_index, edge_weight, x.size(self.node_dim), False,
                            add_self_loops=self.add_self_loops_l1, dtype=x.dtype)
                        self.check_inc(edge_index=edge_index_C, inc=incident_matrix, normalize=True)

                    if self.cached:
                        self._cached_inc = incident_matrix
                        self.init_z = torch.zeros((incident_matrix.sizes()[0], x.size()[-1])).cuda()
                else:
                    incident_matrix = self._cached_inc

                cache = self._cached_adj_t
                if cache is None:
                    # if True:
                    if False:
                        edge_index = self.doubly_stochastic_norm(edge_index, x, self.add_self_loops)  ##
                    else:
                        edge_index = gcn_norm(
                            edge_index, edge_weight, x.size(self.node_dim), False,
                            add_self_loops=self.add_self_loops, dtype=x.dtype)

                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        K_ = self.K if niter is None else niter
        assert edge_weight is None
        if K_ <= 0:
            return x

        hh = x

        x, xs = self.gtn_forward(x=x, hh=hh, incident_matrix=incident_matrix, K=K_)
        return x, xs

    def gtn_forward(self, x, hh, K, incident_matrix):
        lambda2 = self.lambda2
        beta = self.beta
        gamma = None

        ############################# parameter setting ##########################
        if gamma is None:
            gamma = 1

        if beta is None:
            beta = 1 / 2

        if lambda2 > 0: z = self.init_z.detach()

        xs = []
        for k in range(K):
            grad = x - hh
            smoo = x - gamma * grad
            temp = z + beta / gamma * (incident_matrix @ (smoo - gamma * (incident_matrix.t() @ z)))

            z = self.proximal_l1_conjugate(x=temp, lambda2=lambda2, beta=beta, gamma=gamma, m="L1")
            # import ipdb; ipdb.set_trace()

            ctz = incident_matrix.t() @ z

            x = smoo - gamma * ctz

            x = F.dropout(x, p=self.dropout, training=self.training)

        # print("wihtout average")
        light_out = x

        return light_out, xs

    def proximal_l1_conjugate(self, x: Tensor, lambda2, beta, gamma, m):
        if m == 'L1':
            x_pre = x
            x = torch.clamp(x, min=-lambda2, max=lambda2)
            # print('diff after proximal: ', (x-x_pre).norm())

        elif m == 'L1_original':  ## through conjugate
            rr = gamma / beta
            yy = rr * x
            x_pre = x
            temp = torch.sign(yy) * torch.clamp(torch.abs(yy) - rr * lambda2, min=0)
            x = x - temp / rr

        else:
            raise ValueError('wrong prox')
        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, alpha={}, dropout={})'.format(self.__class__.__name__, self.K,
                                                                self.alpha, self.dropout)


class BipartiteNorm(torch.nn.Module):
    def __init__(self, num_features: int, num_users: int) -> None:
        super().__init__()

        self.UserNorm = torch.nn.BatchNorm1d(num_features)
        self.ItemNorm = torch.nn.BatchNorm1d(num_features)
        self.num_users = num_users

    def forward(self, x: torch.Tensor):
        users, items = x[:self.num_users], x[self.num_users:]
        users, items = self.UserNorm(users), self.ItemNorm(items)
        return torch.cat((users, items), dim=0)


# For SimpleX, from https://github.com/reczoo/RecZoo/blob/main/matching/cf/SimpleX/src/SimpleX.py
class BehaviorAggregator(nn.Module):
    def __init__(self, embedding_dim, gamma=0.5, aggregator="mean", dropout_rate=0.):
        super(BehaviorAggregator, self).__init__()
        self.aggregator = aggregator
        self.gamma = gamma
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        if self.aggregator in ["user_attention", "self_attention"]:
            self.W_k = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                     nn.Tanh())
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
            if self.aggregator == "self_attention":
                self.W_q = nn.Parameter(torch.Tensor(embedding_dim, 1))
                nn.init.xavier_normal_(self.W_q)

    def forward(self, uid_emb, sequence_emb):
        out = uid_emb
        if self.aggregator == "mean":
            out = self.average_pooling(sequence_emb)
        elif self.aggregator == "user_attention":
            out = self.user_attention(uid_emb, sequence_emb)
        elif self.aggregator == "self_attention":
            out = self.self_attention(sequence_emb)
        return self.gamma * uid_emb + (1 - self.gamma) * out

    def user_attention(self, uid_emb, sequence_emb):
        key = self.W_k(sequence_emb) # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.bmm(key, uid_emb.unsqueeze(-1)).squeeze(-1) # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_v(output)

    def self_attention(self, sequence_emb):
        key = self.W_k(sequence_emb) # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.matmul(key, self.W_q).squeeze(-1) # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_v(output)

    def average_pooling(self, sequence_emb):
        mask = sequence_emb.sum(dim=-1) != 0
        mean = sequence_emb.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1.e-9)
        return self.W_v(mean)

    def masked_softmax(self, X, mask):
        # use the following softmax to avoid nans when a sequence is entirely masked
        X = X.masked_fill_(mask, 0)
        e_X = torch.exp(X)
        return e_X / (e_X.sum(dim=1, keepdim=True) + 1.e-9)
