import os
import gc
import math
import pickle
import random
import numpy as np
from copy import deepcopy
from sklearn.decomposition import PCA
from scipy.special import factorial, comb
from scipy.sparse import coo_matrix, diags, hstack, vstack, identity
from scipy.sparse.linalg import eigsh

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

from .modeling_utils import GeneralPropagation, BipartiteNorm
from .base_model import BaseRec, BaseGCN


class BPR(BaseRec):
    def __init__(self, dataset, args):
        """
        Parameters
        ----------
        num_users : int
        num_users : int
        dim : int
            embedding dimension
        """
        super(BPR, self).__init__(dataset, args)

        # User / Item Embedding
        self.user_emb = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_emb = nn.Embedding(self.num_items, self.embedding_dim)

        self.reset_para()

    def reset_para(self):
        nn.init.normal_(self.user_emb.weight, mean=0., std=self.init_std)
        nn.init.normal_(self.item_emb.weight, mean=0., std=self.init_std)
        
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        """
        Parameters
        ----------
        batch_user : 1-D LongTensor (batch_size)
        batch_pos_item : 1-D LongTensor (batch_size)
        batch_neg_item : 1-D LongTensor (batch_size)

        Returns
        -------
        output : 
            Model output to calculate its loss function
        """
        
        u = self.user_emb(batch_user)
        i = self.item_emb(batch_pos_item)
        j = self.item_emb(batch_neg_item)
        
        pos_score = (u * i).sum(dim=1, keepdim=True)
        neg_score = (u * j).sum(dim=1, keepdim=True)

        return pos_score, neg_score

    def get_loss(self, output):
        """Compute the loss function with the model output

        Parameters
        ----------
        output : 
            model output (results of forward function)

        Returns
        -------
        loss : float
        """
        pos_score, neg_score = output[0], output[1]
        loss = -F.logsigmoid(pos_score - neg_score).sum()
        return loss

    def forward_multi_items(self, batch_user, batch_items):
        """forward when we have multiple items for a user

        Parameters
        ----------
        batch_user : 1-D LongTensor (batch_size)
        batch_items : 2-D LongTensor (batch_size x k)

        Returns
        -------
        score : 2-D FloatTensor (batch_size x k)
        """

        batch_user = batch_user.unsqueeze(-1)
        batch_user = torch.cat(batch_items.size(1) * [batch_user], 1)
        
        u = self.user_emb(batch_user)		# batch_size x k x dim
        i = self.item_emb(batch_items)		# batch_size x k x dim
        
        score = (u * i).sum(dim=-1, keepdim=False)
        
        return score

    def get_user_embedding(self, batch_user):
        return self.user_emb(batch_user)
    
    def get_item_embedding(self, batch_item):
        return self.item_emb(batch_item)

    def get_all_pre_embedding(self):
        """get total embedding of users and items

        Returns
        -------
        users : 2-D FloatTensor (num. users x dim)
        items : 2-D FloatTensor (num. items x dim)
        """
        users = self.user_emb(self.user_list)
        items = self.item_emb(self.item_list)

        return users, items
    
    def get_all_ratings(self):
        users, items = self.get_all_pre_embedding()
        score_mat = torch.matmul(users, items.T)
        return score_mat
    
    def get_ratings(self, batch_user):
        users, items = self.get_all_pre_embedding()
        users = users[batch_user]
        score_mat = torch.matmul(users, items.T)
        return score_mat


class LightGCN(BaseGCN):
    def __init__(self, dataset, args):
        super(LightGCN, self).__init__(dataset, args)
        
        self.embedding_dim = args.embedding_dim
        self.num_layers = args.num_layers
        self.keep_prob = getattr(args, "keep_prob", 0.)
        self.A_split = getattr(args, "A_split", False)
        self.dropout = getattr(args, "dropout", False)
        self.init_std = args.init_std

        self.user_emb = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.item_emb = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
        
        self.Graph = self.dataset.SparseGraph

        self.aug_graph = getattr(args, "aug_graph", False)
        if self.aug_graph:
            self.augGraph = pickle.load(open(os.path.join("data", args.dataset, "aug_graph.pkl"), "rb")).cuda()
        else:
            self.augGraph = None

        self.reset_para()
    
    def reset_para(self):
        nn.init.normal_(self.user_emb.weight, std=self.init_std)
        nn.init.normal_(self.item_emb.weight, std=self.init_std)

    def _dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse_coo_tensor(index.t(), values, size, dtype=torch.float32)
        return g.coalesce()

    def _dropout(self, keep_prob, Graph):
        if self.A_split:
            graph = []
            for g in Graph:
                graph.append(self._dropout_x(g, keep_prob))
        else:
            graph = self._dropout_x(Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        if self.training and self.aug_graph:
            Graph = self.augGraph
        else:
            Graph = self.Graph
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb, items_emb])
        light_out = all_emb
        if self.dropout:
            if self.training:
                g_droped = self._dropout(self.keep_prob, Graph)
            else:
                g_droped = Graph
        else:
            g_droped = Graph

        for layer in range(1, self.num_layers + 1):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            light_out = (light_out * layer + all_emb) / (layer + 1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def get_all_pre_embedding(self):
        users = self.user_emb(self.user_list)
        items = self.item_emb(self.item_list)
        
        return users, items


# based on https://github.com/liu-jc/PyTorch_NGCF/blob/master/NGCF/Models.py
class NGCF(LightGCN):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()
        self.weight_size = [self.embedding_dim] * self.num_layers
        self.weight_size = [self.embedding_dim] + self.weight_size
        dropout_list = [args.dropout_rate] * self.num_layers
        for i in range(self.num_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            self.dropout_list.append(nn.Dropout(dropout_list[i]))

    def computer(self):
        """
        propagate methods for NGCF
        """
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        ego_embeddings = torch.cat([users_emb, items_emb])
        ngcf_out = [ego_embeddings]
        if self.dropout:
            if self.training:
                g_droped = self._dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        
        for i in range(self.num_layers):
            side_embeddings = torch.sparse.mm(g_droped, ego_embeddings)
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = self.dropout_list[i](ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            ngcf_out += [norm_embeddings]
        
        all_embeddings = torch.cat(ngcf_out, dim=1)
        users, items = torch.split(all_embeddings, [self.num_users, self.num_items])
        return users, items


class UltraGCN(BPR):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)

        self.w1 = args.ultra_w1
        self.w2 = args.ultra_w2
        self.w3 = args.ultra_w3
        self.w4 = args.ultra_w4
        self.negative_weight = args.w_negative
        self.gamma = args.w_norm
        self.lambda_ = args.w_ii
        self.ii_neighbor_num = args.ultra_ii_num

        self.constraint_mat = self.construct_ui_matrix()
        self.ii_neighbor_mat, self.ii_constraint_mat = self.construct_ii_matrix()

    def construct_ui_matrix(self):
        # construct degree matrix for graphmf
        train_mat = self.dataset.train_mat
        items_D = torch.sparse.sum(train_mat, dim=0).to_dense()
        users_D = torch.sparse.sum(train_mat, dim=1).to_dense()

        beta_uD = (torch.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / torch.sqrt(items_D + 1)).reshape(1, -1)

        constraint_mat = {"beta_uD": beta_uD.reshape(-1).cuda(),
                        "beta_iD": beta_iD.reshape(-1).cuda()}
        return constraint_mat
    
    def construct_ii_matrix(self):
        f_ii_neighbor_mat = os.path.join("data", self.args.dataset, "ultragcn_ii_neighbor_mat.pkl")
        f_ii_constraint_mat = os.path.join("data", self.args.dataset, "ultragcn_ii_constraint_mat.pkl")
        if os.path.exists(f_ii_constraint_mat):
            f_ii_neighbor_mat = pickle.load(open(f_ii_neighbor_mat, "rb"))
            ii_constraint_mat = pickle.load(open(f_ii_constraint_mat, "rb"))
            return f_ii_neighbor_mat.cuda(), ii_constraint_mat.cuda()
        
        print('Computing \\Omega for the item-item graph... ')
        train_mat = self.dataset.train_mat.to(torch.float)
        num_neighbors = self.ii_neighbor_num
        A = torch.sparse.mm(train_mat.T, train_mat)	# I * I
        n_items = A.shape[0]
        res_mat = torch.zeros((n_items, num_neighbors))
        res_sim_mat = torch.zeros((n_items, num_neighbors))

        items_D = torch.sparse.sum(A, dim=0).to_dense()
        users_D = torch.sparse.sum(A, dim=1).to_dense()

        beta_uD = (torch.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / torch.sqrt(items_D + 1)).reshape(1, -1)
        all_ii_constraint_mat = beta_uD.mm(beta_iD)
        for i in range(n_items):
            row = all_ii_constraint_mat[i] * A[i].to_dense()
            row_sims, row_idxs = torch.topk(row, num_neighbors)
            res_mat[i] = row_idxs
            res_sim_mat[i] = row_sims
            if i % 15000 == 0:
                print('i-i constraint matrix {} ok'.format(i))

        print('Computation \\Omega OK!')
        pickle.dump(res_mat.long(), open(f_ii_neighbor_mat, "wb"))
        pickle.dump(res_sim_mat.float(), open(f_ii_constraint_mat, "wb"))
        return res_mat.long().cuda(), res_sim_mat.float().cuda()

    def get_device(self):
        return self.user_emb.weight.device

    def get_omegas(self, users, pos_items, neg_items):
        device = self.get_device()
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).to(device)
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)
        
        if self.w4 > 0:
            neg_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][neg_items]).to(device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(len(neg_items)).to(device)

        weight = torch.cat((pos_weight, neg_weight))
        return weight

    def cal_loss_L(self, pos_scores, neg_scores, omega_weight):
        device = self.get_device()
        
        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight=omega_weight[len(pos_scores):], reduction='none')
        
        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight=omega_weight[:len(pos_scores)], reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight
      
        return loss.sum()

    def cal_loss_I(self, users, pos_items):
        device = self.get_device()
        neighbor_embeds = self.item_emb(self.ii_neighbor_mat[pos_items].to(device))    # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].to(device)     # len(pos_items) * num_neighbors
        user_embeds = self.user_emb(users).unsqueeze(1)
        
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
      
        # loss = loss.sum(-1)
        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2
    
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        """
        Parameters
        ----------
        batch_user : 1-D LongTensor (batch_size)
        batch_pos_item : 1-D LongTensor (batch_size)
        batch_neg_item : 1-D LongTensor (batch_size)

        Returns
        -------
        output : 
            Model output to calculate its loss function
        """
        user_embeds = self.user_emb(batch_user)
        pos_embeds = self.item_emb(batch_pos_item)
        neg_embeds = self.item_emb(batch_neg_item)
      
        pos_score = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_score = (user_embeds * neg_embeds).sum(dim=-1) # batch_size
        return pos_score, neg_score, batch_user, batch_pos_item, batch_neg_item

    def get_loss(self, output):
        """Compute the loss function with the model output

        Parameters
        ----------
        output : 
            model output (results of forward function)

        Returns
        -------
        loss : float
        """
        pos_score, neg_score, batch_user, batch_pos_item, batch_neg_item = output
        omega_weight = self.get_omegas(batch_user, batch_pos_item, batch_neg_item)
        loss = self.cal_loss_L(pos_score, neg_score, omega_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(batch_user, batch_pos_item)
        return loss
    

class JGCF(LightGCN):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.alpha = args.jgcf_alpha
        self.a = args.jgcf_jacobi_a
        self.b = args.jgcf_jacobi_b
        self.base_coeff = args.base_coeff

    def JacobiConv(self, L, xs, adj, coeffs, l=-1.0, r=1.0):
        '''
        Jacobi Bases.
        '''
        a = self.a
        b = self.b
        if L == 0: return xs[0]
        if L == 1:
            coef1 = (a - b) / 2 - (a + b + 2) / 2 * (l + r) / (r - l)
            coef1 *= coeffs[0]
            coef2 = (a + b + 2) / (r - l)
            coef2 *= coeffs[0]
            return coef1 * xs[-1] + coef2 * torch.sparse.mm(adj, xs[-1])
        coef_l = 2 * L * (L + a + b) * (2 * L - 2 + a + b)
        coef_lm1_1 = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
        coef_lm1_2 = (2 * L + a + b - 1) * (a**2 - b**2)
        coef_lm2 = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)
        tmp1 = coeffs[L - 1] * (coef_lm1_1 / coef_l)
        tmp2 = coeffs[L - 1] * (coef_lm1_2 / coef_l)
        tmp3 = coeffs[L - 1] * coeffs[L - 2] * (coef_lm2 / coef_l)
        tmp1_2 = tmp1 * (2 / (r - l))
        tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2
        nx = tmp1_2 * torch.sparse.mm(adj, xs[-1]) - tmp2_2 * xs[-1]
        nx -= tmp3 * xs[-2]
        return nx

    def computer(self):
        """
        propagate methods for JGCF
        """
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb, items_emb])
        coeffs = nn.ParameterList([
            nn.Parameter(torch.tensor(float(min(1 / self.base_coeff, 1))), requires_grad=False) for i in range(self.num_layers + 1)
        ])
        coeffs = [self.base_coeff * torch.tanh(_) for _ in coeffs]
        jgcf_out = [self.JacobiConv(0, [all_emb], self.Graph, coeffs)]
        for layer in range(1, self.num_layers + 1):
            tx = self.JacobiConv(layer, jgcf_out, self.Graph, coeffs)
            jgcf_out.append(tx)
        jgcf_out = [x.unsqueeze(1) for x in jgcf_out]
        all_embeddings_low = torch.cat(jgcf_out, dim=1).mean(1)
        all_embeddings_mid = self.alpha * all_emb - all_embeddings_low
        all_embeddings = torch.hstack([all_embeddings_low, all_embeddings_mid])
        users, items = torch.split(all_embeddings, [self.num_users, self.num_items])
        return users, items


class CAGCN(LightGCN):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.cagcn_type = args.cagcn_type
        self.trend_coeff = args.cagcn_trend_coeff

        if self.cagcn_type == "sc":
            cal_trend_func = self.co_ratio_deg_user_sc
        else:
            raise(NotImplementedError, 'Invalid CAGCN type.')
        edge_index, edge_weight = self.Graph._indices(), self.Graph._values()
        trend = cal_trend_func()
        self.Graph = self.trend_coeff * trend + edge_weight

    def co_ratio_deg_user_sc(self):
        file_name = os.path.join("data", self.args.dataset, "co_ratio_edge_weight_sc.pkl")
        if os.path.exists(file_name):
            edge_weight = pickle.load(open(file_name, "rb"))
            return edge_weight
        
        user_item_graph = self.dataset.train_mat.to_dense()

        edge_weight = torch.zeros((self.num_users + self.num_items, self.num_users + self.num_items))

        for i in range(self.num_items):
            users = user_item_graph[:, i].nonzero().squeeze(-1)

            items = user_item_graph[users]
            user_user_cap = torch.matmul(items, items.t())

            sc_simi = (user_user_cap / ((items.sum(dim=1) * items.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)

            edge_weight[users, i + self.num_users] = sc_simi

        for i in range(self.num_users):
            items = user_item_graph[i, :].nonzero().squeeze(-1)

            users = user_item_graph[:, items].t()
            item_item_cap = torch.matmul(users, users.t())

            sc_simi = (item_item_cap / ((users.sum(dim=1) *
                                        users.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)

            edge_weight[items + self.num_users, i] = sc_simi

        edge_weight = edge_weight / edge_weight.sum(-1, keepdim=True)
        edge_weight[edge_weight <= 1e-9] = 0.
        index = edge_weight.nonzero(as_tuple=False)
        data = edge_weight[edge_weight > 0.]
        trend = torch.sparse_coo_tensor(index.t(), data, torch.Size([self.num_users + self.num_items, self.num_users + self.num_items]), dtype=torch.float)

        pickle.dump(trend, open(file_name, "wb"))

        return trend


class GDE(BaseGCN):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.dataset = dataset
        self.args = args

        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items

        self.user_list = torch.LongTensor([i for i in range(self.num_users)]).cuda()
        self.item_list = torch.LongTensor([i for i in range(self.num_items)]).cuda()

        self.embedding_dim = args.embedding_dim
        self.init_std = args.init_std

        self.user_emb = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.item_emb = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
        
        self.smooth_ratio = args.smooth_ratio
        self.rough_ratio = args.rough_ratio
        self.beta = args.gde_beta
        self.dropout_rate = args.dropout_rate
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.Graph_u, self.Graph_i = self.cal_filterd_graph()

        self.reset_para()

    def reset_para(self):
        nn.init.normal_(self.user_emb.weight, std=self.init_std)
        nn.init.normal_(self.item_emb.weight, std=self.init_std)

    def weight_feature(self, value):
        return torch.exp(self.beta * value)

    def _cal_spectral_feature(self, Adj, size, adj_type='user', smooth='smooth', niter=5):
        if smooth == 'smooth':
            largest = True
        elif smooth == 'rough':
            largest = False
        else:
            raise NotImplementedError
        
        # vector_file_name = os.path.join("data", self.args.dataset, "GDE", f"{smooth}_{adj_type}_{size}_vectors.pkl")
        # value_file_name = os.path.join("data", self.args.dataset, "GDE", f"{smooth}_{adj_type}_{size}_values.pkl")
        
        # if os.path.exists(vector_file_name):
        #     vectors = pickle.load(open(vector_file_name, "rb"))
        #     values = pickle.load(open(value_file_name, "rb"))
        #     return values, vectors
        
        ## torch.logpcg effects the random state
        values, vectors = torch.lobpcg(Adj, k=size, largest=largest, niter=niter)
        # os.makedirs(os.path.dirname(vector_file_name), exist_ok=True)
        # pickle.dump(values, open(value_file_name, "wb"))
        # pickle.dump(vectors, open(vector_file_name, "wb"))
        return values, vectors

    def cal_filterd_graph(self):
        rate_matrix = self.dataset.train_mat.to_dense()
        D_u = rate_matrix.sum(1)
        D_i = rate_matrix.sum(0)
        D_u[D_u == 0.] = 1.
        D_i[D_i == 0.] = 1.
        D_u_sqrt = torch.sqrt(D_u).unsqueeze(dim=1)
        D_i_sqrt = torch.sqrt(D_i).unsqueeze(dim=0)
        rate_matrix = rate_matrix / D_u_sqrt / D_i_sqrt

        del D_u, D_i 
        gc.collect()
        torch.cuda.empty_cache()

        # user-user matrix
        L_u = rate_matrix.mm(rate_matrix.t())

        # smoothed feautes for user-user relations
        smooth_user_values, smooth_user_vectors = self._cal_spectral_feature(L_u, int(self.smooth_ratio * self.num_users), adj_type='user', smooth='smooth')
        # rough feautes for user-user relations
        if self.rough_ratio != 0:
            rough_user_values, rough_user_vectors = self._cal_spectral_feature(L_u, int(self.rough_ratio * self.num_users), adj_type='user', smooth='rough')
        else:
            rough_user_values, rough_user_vectors = torch.tensor([]), torch.tensor([])
        
        # item-item matrix
        L_i = rate_matrix.t().mm(rate_matrix)

        # smoothed feautes for item-item relations
        smooth_item_values, smooth_item_vectors = self._cal_spectral_feature(L_i, int(self.smooth_ratio * self.num_items), adj_type='item', smooth='smooth')
        # rough feautes for item-item relations
        if self.rough_ratio != 0:
            rough_item_values, rough_item_vectors = self._cal_spectral_feature(L_i, int(self.rough_ratio * self.num_items), adj_type='item', smooth='rough')
        else:
            rough_item_values, rough_item_vectors = torch.tensor([]), torch.tensor([])
        
        user_filter = torch.cat([self.weight_feature(smooth_user_values), self.weight_feature(rough_user_values)])
        item_filter = torch.cat([self.weight_feature(smooth_item_values), self.weight_feature(rough_item_values)])

        user_vector = torch.cat([smooth_user_vectors, rough_user_vectors], dim=1)
        item_vector = torch.cat([smooth_item_vectors, rough_item_vectors], dim=1)

        L_u = (user_vector * user_filter).mm(user_vector.t())
        L_i = (item_vector * item_filter).mm(item_vector.t())
        return L_u.cuda(), L_i.cuda()
    
    def computer(self):
        """
        propagate methods for GDE
        """
        Graph_u = self.dropout(self.Graph_u) * (1 - self.dropout_rate)
        Graph_i = self.dropout(self.Graph_i) * (1 - self.dropout_rate)
        users = Graph_u.mm(self.user_emb.weight)
        items = Graph_i.mm(self.item_emb.weight)
        return users, items
    
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        all_users, all_items = self.computer()

        u = all_users[batch_user]
        i = all_items[batch_pos_item]
        j = all_items[batch_neg_item]
        
        pos_score = (u * i).sum(dim=1, keepdim=True)
        neg_score = (u * j).sum(dim=1, keepdim=True)

        neg_weight = (1. - (1. - neg_score.sigmoid().clamp(max=0.99)).log10()).detach()
        neg_score = neg_weight * neg_score
        return pos_score, neg_score

    def get_all_pre_embedding(self):
        users = self.user_emb(self.user_list)
        items = self.item_emb(self.item_list)
        
        return users, items


class PGSP(BaseRec):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)

        self.eig_dim = args.eig_dim
        self.phi = args.pgsp_phi

        ideal_filterd_fea, linear_filterd_fea = self.cal_filterd_features()
        self.score_mat = self.phi * linear_filterd_fea + (1. - self.phi) * ideal_filterd_fea
        self.score_mat = self.score_mat.cuda()

    def cal_filterd_features(self):
        f_ideal_filterd_fea = os.path.join("data", self.args.dataset, "PSGP", f"ideal_filterd_fea_{self.eig_dim}.pkl")
        f_linear_filterd_fea = os.path.join("data", self.args.dataset, "PSGP", f"linear_filterd_fea_{self.eig_dim}.pkl")
        f_vectors = os.path.join("data", self.args.dataset, "PSGP", f"vectors_{self.eig_dim}.pkl")
        f_values = os.path.join("data", self.args.dataset, "PSGP", f"values_{self.eig_dim}.pkl")
        if os.path.exists(f_ideal_filterd_fea):
            ideal_filterd_fea = pickle.load(open(f_ideal_filterd_fea, "rb"))
            linear_filterd_fea = pickle.load(open(f_linear_filterd_fea, "rb"))
            return ideal_filterd_fea, linear_filterd_fea
        
        os.makedirs(os.path.dirname(f_ideal_filterd_fea), exist_ok=True)
        
        R_ui = self.dataset.train_mat.to_dense().cpu().numpy()
        R_ui = coo_matrix(R_ui)     # N, M
        D_u = R_ui.sum(axis=1).T.A[0]
        D_u[D_u == 0.] = 1.
        D_u_inv = diags(np.power(D_u, -1/2), offsets=0) # N, N
        D_i = R_ui.sum(axis=0).A[0]
        D_i[D_i == 0.] = 1.
        D_i_inv = diags(np.power(D_i, -1/2), offsets=0) # M, M
        D_u = diags(np.power(R_ui.sum(axis=1).T.A[0], 1/2), offsets=0)  # N, N
        D_i = diags(np.power(R_ui.sum(axis=0).A[0], 1/2), offsets=0)    # M, M

        R_normu = D_u_inv * R_ui        # N, M
        R_normi = R_ui * D_i_inv        # N, M
        S_ui = D_u_inv * R_ui * D_i_inv    # N, M

        R_uu = R_normi * R_normi.T      # N, N
        R_ii = R_normu.T * R_normu      # M, M

        S_u = S_ui * S_ui.T   # N, N
        S_i = S_ui.T * S_ui   # M, M

        Adj = vstack([hstack([R_uu, R_ui]), hstack([R_ui.T, R_ii])])    # N+M, N+M
        I = identity(self.num_users + self.num_items, dtype=np.float32) # N+M, N+M
        D = Adj.sum(axis=0).A[0]
        D[D == 0.] = 1.
        D_inv = diags(np.power(D, -1/2), offsets=0)     # N+M, N+M
        Adj_norm = D_inv * Adj * D_inv      # N+M, N+M
        L_norm = I - Adj_norm       # N+M, N+M
        
        if os.path.exists(f_vectors):
            values = pickle.load(open(f_values, "rb"))
            vectors = pickle.load(open(f_vectors, "rb"))    # N+M, K
        else:
            values, vectors = eigsh(L_norm, k=self.eig_dim, which='SA')
            # values, vectors = cal_spectral_feature(convert_sp_mat_to_sp_tensor(L_norm), k=self.eig_dim, largest=False).cpu().numpy()
            pickle.dump(values, open(f_values, "wb"))
            pickle.dump(vectors, open(f_vectors, "wb"))
        
        fea_u = hstack([S_u, R_ui])     # N, N+M
        D_fea = fea_u.sum(axis=0).A[0]
        D_fea[D_fea == 0.] = 1.
        D_fea_u_inv = diags(np.power(D_fea, -1/2), offsets=0)  # N+M, N+M
        D_fea_u = diags(np.power(D_fea, 1/2), offsets=0)   # N+M, N+M
        D_fea_u = D_fea_u.toarray()

        linear_filterd_fea = R_ui * D_i_inv * S_i * D_i     # N, M
        linear_filterd_fea = linear_filterd_fea.toarray()
        # A = vstack([hstack([S_u, S_ui]), hstack([S_ui.T, S_i])])    # N+M, N+M
        # spec_linear_filterd_fea = (fea_u * D_fea_u_inv * A).toarray()
        # linear_filterd_fea = np.matmul(spec_linear_filterd_fea, D_fea_u)	# N, N+M
        # linear_filterd_fea = linear_filterd_fea[:, self.num_users:]   # N, M
        linear_filterd_fea = torch.from_numpy(linear_filterd_fea)
        pickle.dump(linear_filterd_fea, open(f_linear_filterd_fea, "wb"))

        spec_ideal_filterd_fea = fea_u * D_fea_u_inv * vectors      # N, K
        ideal_filterd_fea = np.matmul(spec_ideal_filterd_fea, np.matmul(vectors.T, D_fea_u))    # N, N+M
        ideal_filterd_fea = ideal_filterd_fea[:, self.num_users:]   # N, M
        ideal_filterd_fea = torch.from_numpy(ideal_filterd_fea)
        pickle.dump(ideal_filterd_fea, open(f_ideal_filterd_fea, "wb"))

        return ideal_filterd_fea, linear_filterd_fea
    
    def forward_multi_items(self, batch_user, batch_items):
        """forward when we have multiple items for each user

        Parameters
        ----------
        batch_user : 1-D LongTensor (batch_size)
        batch_items : 2-D LongTensor (batch_size x k)

        Returns
        -------
        score : 2-D FloatTensor (batch_size x k)
        """
        batch_user = batch_user.unsqueeze(-1)
        batch_user = torch.cat(batch_items.size(1) * [batch_user], 1)
        
        score = self.score_mat[batch_user, batch_items]
        
        return score

    def get_all_ratings(self):
        return self.score_mat

    def get_ratings(self, batch_user):
        return self.score_mat[batch_user]


class SVD_GCN(BaseGCN):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)

        self.svdgcn_type = args.svdgcn_type
        self.svd_dim = args.svd_dim
        self.alpha = args.svdgcn_alpha
        self.beta = args.svdgcn_beta
        if self.svdgcn_type == "nonparametric":
            args.epochs = 0
        elif self.svdgcn_type == "parametric":
            self.embedding_dim = args.embedding_dim
            self.FS = torch.nn.Parameter(
                torch.nn.init.uniform_(
                    torch.randn(self.svd_dim, self.embedding_dim), 
                    -np.sqrt(6. / (self.svd_dim + self.embedding_dim)), 
                    np.sqrt(6. / (self.svd_dim + self.embedding_dim))
                ), requires_grad=True
            )
        else:
            raise NotImplementedError
        
        self.user_vector, self.item_vector = self.cal_ui_vectors()

    def weight_func(self, x):
        return torch.exp(self.beta * x)

    def cal_ui_vectors(self):
        train_mat = self.dataset.train_mat.to_dense()
        D_u = train_mat.sum(1) + self.alpha
        D_i = train_mat.sum(0) + self.alpha
        D_u[D_u == 0.] = 1.
        D_i[D_i == 0.] = 1.
        D_u_sqrt = torch.sqrt(D_u).unsqueeze(dim=1)
        D_i_sqrt = torch.sqrt(D_i).unsqueeze(dim=0)
        train_mat = train_mat / D_u_sqrt / D_i_sqrt
        f_U = os.path.join("data", self.args.dataset, "SVD_GCN", f"svd_U_alpha_{self.alpha}.pkl")
        f_values = os.path.join("data", self.args.dataset, "SVD_GCN", f"svd_values_alpha_{self.alpha}.pkl")
        f_V = os.path.join("data", self.args.dataset, "SVD_GCN", f"svd_V_alpha_{self.alpha}.pkl")
        if os.path.exists(f_V):
            U = pickle.load(open(f_U, "rb"))
            values = pickle.load(open(f_values, "rb"))
            V = pickle.load(open(f_V, "rb"))
        else:
            U, values, V = torch.svd_lowrank(train_mat, q=1000, niter=30)
            os.makedirs(os.path.dirname(f_V), exist_ok=True)
            pickle.dump(U, open(f_U, "wb"))
            pickle.dump(values, open(f_values, "wb"))
            pickle.dump(V, open(f_V, "wb"))
        
        svd_filter = self.weight_func(values[:self.svd_dim])
        user_vector = U[:, :self.svd_dim] * svd_filter
        item_vector = V[:, :self.svd_dim] * svd_filter
        return user_vector.cuda(), item_vector.cuda()
    
    def computer(self):
        if self.svdgcn_type == "nonparametric":
            users = self.user_vector
            items = self.item_vector
        elif self.svdgcn_type == "parametric":
            users = self.user_vector.mm(self.FS)
            items = self.item_vector.mm(self.FS)
        else:
            raise NotImplementedError
        return users, items
    
    def get_all_pre_embedding(self):
        users = self.user_vector
        items = self.item_vector
        return users, items
    

class GTN(LightGCN):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.K = args.gtn_K
        self.alpha = args.gtn_alpha
        self.lambda2 = args.gtn_lambda2
        self.beta = args.gtn_beta
        self.dropout_rate = args.dropout_rate
        self.gp = GeneralPropagation(
            K=self.K,
            alpha=self.alpha,
            dropout=self.dropout_rate,
            lambda2=self.lambda2,
            beta=self.beta,
            cached=True)
    
    def computer(self):
        """
        propagate methods for GTN
        """
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb, items_emb])
        light_out = all_emb
        if self.dropout:
            if self.training:
                g_droped = self._dropout(self.keep_prob, self.Graph)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        x = all_emb
        rc = g_droped.indices()
        r = rc[0]
        c = rc[1]
        num_nodes = g_droped.shape[0]
        edge_index = SparseTensor(row=r, col=c, value=g_droped.values(), sparse_sizes=(num_nodes, num_nodes))
        emb, embs = self.gp.forward(x, edge_index)
        light_out = emb

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items


class ApeGNN(LightGCN):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.ape_type = args.apegnn_type

        self.user_t = nn.init.xavier_uniform_(torch.empty(self.num_users, 1))
        self.item_t = nn.init.xavier_uniform_(torch.empty(self.num_items, 1))
        self.user_t = nn.Parameter(self.user_t, requires_grad=True)
        self.item_t = nn.Parameter(self.item_t, requires_grad=True)
        t_u = torch.sigmoid(torch.log(dataset.user_pop) + 1e-7)
        t_i = torch.sigmoid(torch.log(dataset.item_pop) + 1e-7)
        self.user_t.data = t_u.reshape(-1, 1)
        self.item_t.data = t_i.reshape(-1, 1)

    def _cal_layer_weight(self, init_weight, layer):
        if self.ape_type == "HT":
            return torch.exp(-init_weight) * torch.pow(init_weight, layer).cuda() / torch.FloatTensor([factorial(layer)]).cuda()
        elif self.ape_type == "APPNP":
            return init_weight * torch.pow(1. - init_weight, layer).cuda()
        else:
            raise NotImplementedError

    def computer(self):
        """
        propagate methods for ApeGNN
        """
        Graph = self.Graph
        user_embed = self.user_emb.weight
        item_embed = self.item_emb.weight
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed

        u_weight = self._cal_layer_weight(self.user_t, 0)
        i_weight = self._cal_layer_weight(self.item_t, 0)

        ego_embeddings = torch.cat([u_weight * user_embed, i_weight * item_embed], dim=0)
        embs = [ego_embeddings]

        if self.dropout:
            if self.training:
                g_droped = self._dropout(self.keep_prob, Graph)
            else:
                g_droped = Graph
        else:
            g_droped = Graph

        for layer in range(1, self.num_layers + 1):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], agg_embed))
                side_embeddings = torch.cat(temp_emb, dim=0)
            else:
                side_embeddings = torch.sparse.mm(g_droped, agg_embed)
            
            user_embedds, item_embedds = torch.split(side_embeddings, [self.num_users, self.num_items], dim=0)
            user_embedds = user_embedds * self._cal_layer_weight(self.user_t, layer)
            item_embedds = item_embedds * self._cal_layer_weight(self.item_t, layer)
            side_embeddings_cur = torch.cat([user_embedds, item_embedds], dim=0)
            agg_embed = side_embeddings
            embs.append(side_embeddings_cur)
        embs = torch.stack(embs, dim=1)
        embs = embs.sum(dim=1)
        return embs[:self.num_users, :], embs[self.num_users:, :]


class LGCN(LightGCN):
    # TODO: compute reg only for embeds (not for kernels)
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.frequency = args.lgcn_num_freq
        self.kernels = nn.ParameterList([nn.Parameter(torch.zeros(self.frequency)) for l in range(self.num_layers)])
        for kernel in self.kernels:
            nn.init.normal_(kernel, mean=0.01, std=0.02)
        self.layer_weight = [(1 / (l + 1)) ** 1 for l in range(self.num_layers + 1)]

        values, self.vectors = self._cal_spectral_feature(self.Graph, k=self.frequency)

    def _cal_spectral_feature(self, Adj, k, niter=5):
        f_vector = os.path.join("data", self.args.dataset, "LGCN", f"vectors_{k}.pkl")
        f_value = os.path.join("data", self.args.dataset, "LGCN", f"values_{k}.pkl")
        
        if os.path.exists(f_vector):
            vectors = pickle.load(open(f_vector, "rb"))
            values = pickle.load(open(f_value, "rb"))
            return values.cuda(), vectors.cuda()
        
        values, vectors = torch.lobpcg(Adj, k=k, largest=True, niter=niter)
        os.makedirs(os.path.dirname(f_vector), exist_ok=True)
        pickle.dump(values, open(f_value, "wb"))
        pickle.dump(vectors, open(f_vector, "wb"))
        return values.cuda(), vectors.cuda()
    
    def computer(self):
        """
        propagate methods for LGCN
        """
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb, items_emb])
        lgcn_out = self.layer_weight[0] * all_emb

        for layer in range(1, self.num_layers + 1):
            all_emb = self.vectors.matmul(torch.diag(self.kernels[layer - 1])).matmul(self.vectors.T).matmul(all_emb)
            lgcn_out += self.layer_weight[layer] * all_emb
        
        users, items = torch.split(lgcn_out, [self.num_users, self.num_items])
        return users, items
    

class GFCF(BaseRec):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        
        self.alpha = args.gfcf_alpha
        self.eig_dim = args.eig_dim
        self.adj_mat = self.dataset.train_mat.cuda()
        rowsum = torch.sparse.sum(self.adj_mat, dim=1).to_dense().reshape(-1, 1)
        Du_inv = torch.pow(rowsum, -0.5)
        Du_inv[torch.isinf(Du_inv)] = 0.

        colsum = torch.sparse.sum(self.adj_mat, dim=0).to_dense().reshape(1, -1)
        self.Di = torch.pow(colsum, 0.5)
        self.Di_inv = torch.pow(colsum, -0.5)
        self.Di_inv[torch.isinf(self.Di_inv)] = 0.
        self.norm_adj = self.adj_mat * Du_inv * self.Di_inv
        s, vt = self._cal_spectral_feature(self.norm_adj, self.eig_dim)
        self.score_mat = self.filter(self.adj_mat, self.norm_adj, vt)

    def _cal_spectral_feature(self, Adj, k, niter=30):
        f_vector = os.path.join("data", self.args.dataset, "GFCF", f"vectors_{k}.pkl")
        f_value = os.path.join("data", self.args.dataset, "GFCF", f"values_{k}.pkl")
        
        if os.path.exists(f_vector):
            V = pickle.load(open(f_vector, "rb"))
            values = pickle.load(open(f_value, "rb"))
            return values.cuda(), V.cuda()
        
        U, values, V = torch.svd_lowrank(Adj, q=k, niter=niter)
        # Adj_i = torch.sparse.mm(Adj.T, Adj)
        # values_smooth, V_smooth = torch.lobpcg(Adj_i, k=k, largest=True, niter=niter)
        # values_rough, V_rough = torch.lobpcg(Adj_i, k=32, largest=False, niter=niter)
        # V = torch.cat([V_smooth, V_rough], dim=1)
        # values = torch.cat([values_smooth, values_rough])

        os.makedirs(os.path.dirname(f_vector), exist_ok=True)
        pickle.dump(values, open(f_value, "wb"))
        pickle.dump(V, open(f_vector, "wb"))
        return values.cuda(), V.cuda()
    
    def filter(self, adj_mat, norm_adj, vt):
        adj_mat = adj_mat.float().to_dense()
        U_2 = adj_mat @ norm_adj.T @ norm_adj   # Linear filter
        U_1 = ((adj_mat * self.Di_inv) @ vt @ vt.T) * self.Di   # Ideal filter
        ret = U_2 + self.alpha * U_1
        return ret
    
    def forward_multi_items(self, batch_user, batch_items):
        batch_user = batch_user.unsqueeze(-1)
        batch_user = torch.cat(batch_items.size(1) * [batch_user], 1)
        
        score = self.score_mat[batch_user, batch_items]
        
        return score

    def get_all_ratings(self):
        return self.score_mat

    def get_ratings(self, batch_user):
        return self.score_mat[batch_user]


# https://github.com/MTandHJ/StableGCN/blob/master/main.py
class StableGCN(BaseGCN):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.num_filters = args.svd_dim
        self.dropout_rate = args.dropout_rate
        self.hidden_dim = args.hidden_dim
        self.embedding_dim = args.embedding_dim
        self.alpha = args.stablegcn_alpha
        self.weight = args.stablegcn_lmbda
        self.num_layers = args.num_layers
        self.upper = args.stablegcn_upper
        self.lower = args.stablegcn_lower

        self.Graph = self.dataset.SparseGraph
        self.dense = torch.nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.num_filters, self.hidden_dim, bias=False),
                BipartiteNorm(self.hidden_dim, self.num_users),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                BipartiteNorm(self.hidden_dim, self.num_users),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.embedding_dim, bias=False),
            )
        
        self.user_vectors, self.item_vectors = self.cal_ui_vectors()
        self.initialize()

        self.T_cur = 0
        self.T_max = self.args.epochs * math.ceil(len(dataset) / self.args.batch_size)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def weight_filter(self, vals: torch.Tensor):
        return vals.div(math.sqrt(self.num_filters))
    
    def cal_ui_vectors(self):
        R = self.dataset.train_mat
        userDegs = torch.sparse.sum(R, dim=1).to_dense().reshape(-1, 1) + self.alpha
        itemDegs = torch.sparse.sum(R, dim=0).to_dense().reshape(1, -1) + self.alpha
        userDegs = 1. / torch.sqrt(userDegs)
        itemDegs = 1. / torch.sqrt(itemDegs)
        userDegs[torch.isinf(userDegs)] = 0.
        itemDegs[torch.isinf(itemDegs)] = 0.
        R = userDegs * R * itemDegs
        U, vals, V = torch.svd_lowrank(R, q=self.num_filters, niter=30)
        vals = self.weight_filter(vals)
        user_vecs = U * vals * math.sqrt(U.size(0))
        item_vecs = V * vals * math.sqrt(V.size(0))
        return user_vecs.cuda(), item_vecs.cuda()
    
    def get_all_pre_embedding(self):
        return self.user_vectors, self.item_vectors
    
    def computer(self, return_all=False):
        userEmbs = self.user_vectors
        itemEmbs = self.item_vectors
        embds = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        embds = self.dense(embds)
        features = embds
        avgFeats = embds
        for _ in range(self.num_layers):
            features = torch.sparse.mm(self.Graph, features) * self.weight / (self.weight + 1)
            avgFeats = avgFeats + features
        avgFeats = avgFeats / (self.weight + 1)
        pre_users, pre_items = torch.split(embds, [self.num_users, self.num_items])
        post_users, post_items = torch.split(avgFeats, [self.num_users, self.num_items])
        if return_all:
            return post_users, post_items, pre_users, pre_items
        else:
            return post_users, post_items
    
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        """
        Parameters
        ----------
        batch_user : 1-D LongTensor (batch_size)
        batch_pos_item : 1-D LongTensor (batch_size)
        batch_neg_item : 1-D LongTensor (batch_size)

        Returns
        -------
        output : 
            Model output to calculate its loss function
        """
        post_users, post_items, pre_users, pre_items = self.computer(return_all=True)

        pre_u = pre_users[batch_user]
        pre_i = pre_items[batch_pos_item]
        pre_j = pre_items[batch_neg_item]
        
        pre_pos_score = (pre_u * pre_i).sum(dim=1, keepdim=True)
        pre_neg_score = (pre_u * pre_j).sum(dim=1, keepdim=True)

        post_u = post_users[batch_user]
        post_i = post_items[batch_pos_item]
        post_j = post_items[batch_neg_item]
        
        post_pos_score = (post_u * post_i).sum(dim=1, keepdim=True)
        post_neg_score = (post_u * post_j).sum(dim=1, keepdim=True)

        return pre_pos_score, pre_neg_score, post_pos_score, post_neg_score
    
    def get_loss(self, output):
        pre_pos_score, pre_neg_score, post_pos_score, post_neg_score = output
        loss1 = -F.logsigmoid(post_pos_score - post_neg_score).sum()
        loss2 = -F.logsigmoid(pre_pos_score - pre_neg_score).sum()
        
        weight = self.upper - (self.T_cur / self.T_max) * (self.upper - self.lower)
        self.T_cur += 1

        loss = loss1 * (1. - weight) + loss2 * weight
        return loss
