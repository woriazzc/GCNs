import os
import pickle
import torch
import argparse
import numpy as np
import random
import numpy.linalg as LA
from dataset import implicit_CF_dataset, load_data


# Graph Augmentation Module - GraphAug for implementing p(\widehat{A}|A,X)
# Based on https://github.com/LirongWu/KDGA/blob/main/src/model.py
class GraphAugmentor(torch.nn.Module):
    def __init__(self, alpha, norm=False):
        super().__init__()
        self.alpha = alpha
        self.norm = norm

    def forward(self, h, adj_orig):
        """
        h: node features. torch.Tensor of shape(num_nodes, hidden_dim)
        adj_orig: original adjacency matrix. torch.Tensor of shape(num_nodes, num_nodes)
        """
        N = adj_orig.size(0)
        num_edges = adj_orig.indices().size(1)
        adj_logits = h @ h.T
        edge_probs = torch.sigmoid(adj_logits)
        if self.norm:
            edge_probs = edge_probs / edge_probs.max()
        edge_probs = self.alpha * edge_probs + (1 - self.alpha) * adj_orig
        ids = adj_orig.indices()
        del adj_logits, adj_orig
        torch.cuda.empty_cache()

        edge_probs[edge_probs < 0.5] = 0
        # index = []
        # for i in range(5):
        #     index.append(edge_probs[N // 5 * i:N // 5 * (i + 1)].nonzero(as_tuple=False))
        # index.append(edge_probs[N // 5 * 5:].nonzero(as_tuple=False))
        # index = torch.cat(index, dim=0)
        # index = index[random.sample(range(len(index)), num_edges)]
        edge_probs = edge_probs.cpu()
        index = edge_probs.nonzero(as_tuple=False)
        index = index[random.sample(range(len(index)), num_edges)]
        data = torch.ones(len(index), device=index.device)
        adj_sampled = torch.sparse_coo_tensor(index.t(), data, torch.Size([N, N]), dtype=torch.float)

        del edge_probs
        torch.cuda.empty_cache()
        
        dense = adj_sampled.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        dense = dense.cpu()
        index = dense.nonzero(as_tuple=False)
        data = dense[dense >= 1e-9]
        assert len(index) == len(data)
        adj_sampled = torch.sparse_coo_tensor(index.t(), data, torch.Size([N, N]), dtype=torch.float)
        adj_sampled = adj_sampled.coalesce()
        
        return adj_sampled


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--dataset', type=str, default='gowalla')
parser.add_argument('--model', type=str, default='jgcf')
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--norm', action='store_true')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

if __name__ == '__main__':
    items = np.load(f'crafts/{args.dataset}_{args.model}_post_items.npy')
    users = np.load(f'crafts/{args.dataset}_{args.model}_post_users.npy')
    num_users, num_items, train_pairs, valid_pairs, test_pairs, train_dict, valid_dict, test_dict, train_matrix, user_pop, item_pop = load_data(args.dataset)
    trainset = implicit_CF_dataset(num_users, num_items, train_pairs, train_matrix, train_dict, user_pop, item_pop, 1)
    adj_mat = trainset.oriSparseGraph
    embs = torch.from_numpy(np.concatenate([users, items])).cuda()
    aug = GraphAugmentor(args.alpha, args.norm)
    adj_aug = aug(embs, adj_mat)
    pickle.dump(adj_aug, open(os.path.join("data", args.dataset, "aug_graph.pkl"), "wb"))
