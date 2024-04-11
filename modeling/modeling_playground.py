from .modeling import *
from .modeling_utils import *


class CurriGCN(LightGCN):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.update_interval = args.update_interval
        self.drop_ratio = args.drop_ratio
        self.drop_num = args.drop_num
        self.curr_train_dict = deepcopy(dataset.train_dict)
        self.currGraph = self.Graph
    
    def updateCurrGraph(self):
        candidates = []
        for user, items in self.curr_train_dict.items():
            if len(items) > self.drop_num:
                candidates.append(user)
        if len(candidates) == 0:
            return
        random.shuffle(candidates)
        dropped_users = candidates[:math.ceil(len(candidates) * self.drop_ratio)]
        user_dropped = np.zeros(self.dataset.num_users, dtype=bool)
        user_dropped[dropped_users] = True
        train_pairs = []
        for user in self.curr_train_dict:
            if user_dropped[user]:
                self.curr_train_dict[user] = self.curr_train_dict[user][:-self.drop_num]
            for item in self.curr_train_dict[user]:
                train_pairs.append([user, item])
        train_pairs = torch.tensor(train_pairs)
        self.currGraph = convert_pairs_to_graph(train_pairs, self.num_users, self.num_items)
    
    def do_something_in_each_epoch(self, epoch):
        if epoch % self.update_interval == 0 and epoch > 0:
            self.updateCurrGraph()
    
    def withCurrGraph(func):
        def wrapper(self, *args, **kwargs):
            if self.training:
                oldGraph = deepcopy(self.Graph)
                self.Graph = self.currGraph
                result = func(self, *args,**kwargs)
                self.Graph = oldGraph
                return result
            else:
                return func(self, *args,**kwargs)
        return wrapper
    
    @withCurrGraph
    def computer(self):
        return super().computer()
    

class RandGCN(LightGCN):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.user_emb.weight.requires_grad = False
        self.item_emb.weight.requires_grad = False


class PrepGCN(LightGCN):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.neighborhood = args.neighborhood
        self.user_feature, self.item_feature = self._build_neighborhood_features()
        self.user_feature, self.item_feature = self.user_feature.cuda(), self.item_feature.cuda()
        self.user_feature_dim = self.user_feature.size(-1)
        self.item_feature_dim = self.item_feature.size(-1)
        self.user_embed_layer = nn.Sequential(
            nn.Linear(self.user_feature_dim, self.embedding_dim, bias=False),
            # nn.ReLU(),
            # nn.Dropout(args.dropout_rate),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )
        self.item_embed_layer = nn.Sequential(
            nn.Linear(self.item_feature_dim, self.embedding_dim, bias=False),
            # nn.ReLU(),
            # TODO: ReLU导致变差
            # nn.Dropout(args.dropout_rate),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )
        # nn.init.normal_(self.user_embed_layer.weight, std=self.init_std)
        # nn.init.normal_(self.item_embed_layer.weight, std=self.init_std)

    def _build_neighborhood_features(self):
        neighborhood_features_user = F.one_hot(self.user_list, num_classes=self.num_users).float()
        neighborhood_features_item = F.one_hot(self.item_list, num_classes=self.num_items).float()
        return neighborhood_features_user, neighborhood_features_item
        file_name = os.path.join("data", self.args.dataset, "PrepGCN", f"vectors_512.pkl")
        if os.path.exists(file_name):
            neighborhood_features_PCA = torch.from_numpy(pickle.load(open(file_name, "rb"))).float()
            neighborhood_features_user, neighborhood_features_item = torch.split(neighborhood_features_PCA, [self.num_users, self.num_items])
            return neighborhood_features_user, neighborhood_features_item
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

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
        L_norm = I - Adj_norm

        values, vectors = eigsh(L_norm, k=500, which='SA')
        vectors = torch.from_numpy(vectors)
        pickle.dump(vectors, open(os.path.join("data", self.args.dataset, "PrepGCN", "vectors_500.pkl"), "wb"))
        
        neighborhood_features_user, neighborhood_features_item = torch.split(vectors, [self.num_users, self.num_items])
        return neighborhood_features_user, neighborhood_features_item
    
    def computer(self):
        """
        propagate methods for PrepGCN
        """
        users_emb = self.user_embed_layer(self.user_feature)
        items_emb = self.item_embed_layer(self.item_feature)
        all_emb = torch.cat([users_emb, items_emb])
        light_out = all_emb
        if self.dropout:
            if self.training:
                g_droped = self._dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

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
        users = self.user_embed_layer(self.user_feature)
        items = self.item_embed_layer(self.item_feature)
        
        return users, items


class FreqGCN(LightGCN):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.max_num_smooth = args.freqgcn_max_num_smooth
        self.max_num_rough = args.freqgcn_max_num_rough
        self.T = args.freqgcn_T
        _, self.smooth_vectors = self._cal_spectral_feature(self.Graph, self.max_num_smooth, smooth="smooth")
        _, self.rough_vectors = self._cal_spectral_feature(self.Graph, self.max_num_rough, smooth="rough")
        self.Graph = None
    
    def _cal_spectral_feature(self, Adj, size, smooth='smooth', niter=5):
        if smooth == 'smooth':
            largest = True
        elif smooth == 'rough':
            largest = False
        else:
            raise NotImplementedError
        
        vector_file_name = os.path.join("data", self.args.dataset, "FreqGCN", f"{smooth}_{size}_vectors.pkl")
        value_file_name = os.path.join("data", self.args.dataset, "FreqGCN", f"{smooth}_{size}_values.pkl")
        
        if os.path.exists(vector_file_name):
            vectors = pickle.load(open(vector_file_name, "rb"))
            values = pickle.load(open(value_file_name, "rb"))
            return values, vectors
        
        values, vectors = torch.lobpcg(Adj, k=size, largest=largest, niter=niter)
        os.makedirs(os.path.dirname(vector_file_name), exist_ok=True)
        pickle.dump(values, open(value_file_name, "wb"))
        pickle.dump(vectors, open(vector_file_name, "wb"))
        return values, vectors

    def do_something_in_each_epoch(self, epoch):
        n_rough_vec = min(self.max_num_rough, int(epoch / self.T * self.max_num_rough))
        n_smooth_vec = self.max_num_smooth
        vec = torch.cat([self.smooth_vectors[:, :n_smooth_vec], self.rough_vectors[:, :n_rough_vec]], dim=1)
        self.Graph = torch.matmul(vec, vec.T)
    
    def computer(self):
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb, items_emb])
        gcn_out = torch.matmul(self.Graph, all_emb)
        users, items = torch.split(gcn_out, [self.num_users, self.num_items])
        return users, items


class GFCF_HL(BaseRec):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        
        self.alpha = args.gfcf_alpha
        self.dim_smooth = args.gfcf_hl_dim_smooth
        self.dim_rough = args.gfcf_hl_dim_rough
        self.adj_mat = self.dataset.train_mat.cuda()
        rowsum = torch.sparse.sum(self.adj_mat, dim=1).to_dense().reshape(-1, 1)
        Du_inv = torch.pow(rowsum, -0.5)
        Du_inv[torch.isinf(Du_inv)] = 0.

        colsum = torch.sparse.sum(self.adj_mat, dim=0).to_dense().reshape(1, -1)
        self.Di = torch.pow(colsum, 0.5)
        self.Di_inv = torch.pow(colsum, -0.5)
        self.Di_inv[torch.isinf(self.Di_inv)] = 0.
        self.norm_adj = self.adj_mat * Du_inv * self.Di_inv
        v_smooth, v_rough = self._cal_spectral_feature(self.norm_adj, self.dim_smooth, self.dim_rough)
        self.score_mat = self.filter(self.adj_mat, v_smooth, v_rough)

    def _cal_spectral_feature(self, Adj, k_smooth, k_rough, niter=30):
        Adj_i = torch.sparse.mm(Adj.T, Adj)
        values_smooth, V_smooth = torch.lobpcg(Adj_i, k=k_smooth, largest=True, niter=niter)
        values_rough, V_rough = torch.lobpcg(Adj_i, k=k_rough, largest=False, niter=niter)
        return V_smooth.cuda(), V_rough.cuda()
    
    def filter(self, adj_mat, v_smooth, v_rough):
        adj_mat = adj_mat.float().to_dense()
        U_1 = ((adj_mat * self.Di_inv) @ v_smooth @ v_smooth.T) * self.Di   # low-pass filter
        U_2 = ((adj_mat * self.Di_inv) @ v_rough @ v_rough.T) * self.Di   # high-pass filter
        ret = U_1 + self.alpha * U_2
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


class IdealGCN(LightGCN):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.alpha = args.idealgcn_alpha
        self.dim_smooth = args.idealgcn_dim_smooth
        self.dim_rough = args.idealgcn_dim_rough
        v_smooth, v_rough = self._cal_spectral_feature(self.Graph, self.dim_smooth, self.dim_rough)
        values = torch.cat([torch.ones(self.dim_smooth, dtype=torch.float32),
                            self.alpha * torch.ones(self.dim_rough, dtype=torch.float32)]).cuda()
        vectors = torch.cat([v_smooth, v_rough], dim=1)
        dense = vectors.mm(torch.diag(values)).mm(vectors.T).cpu()
        index = dense.nonzero(as_tuple=False)
        data = dense[index[:, 0], index[:, 1]]
        self.Graph = torch.sparse_coo_tensor(index.t(), data, torch.Size(
            [self.num_users + self.num_items, self.num_users + self.num_items]), dtype=torch.float)
        self.Graph = self.Graph.coalesce().cuda()

    def _cal_spectral_feature(self, Adj, k_smooth, k_rough, niter=30):
        f_smooth = os.path.join("data", self.args.dataset, "IdealGCN", f"V_smooth_{k_smooth}.pkl")
        f_rough = os.path.join("data", self.args.dataset, "IdealGCN", f"V_rough_{k_rough}.pkl")
        if os.path.exists(f_rough) and os.path.exists(f_smooth):
            V_smooth = pickle.load(open(f_smooth, "rb"))
            V_rough = pickle.load(open(f_rough, "rb"))
            return V_smooth.cuda(), V_rough.cuda()
        
        os.makedirs(os.path.dirname(f_rough), exist_ok=True)
        values_smooth, V_smooth = torch.lobpcg(Adj, k=k_smooth, largest=True, niter=niter)
        values_rough, V_rough = torch.lobpcg(Adj, k=k_rough, largest=False, niter=niter)
        pickle.dump(V_smooth, open(f_smooth, "wb"))
        pickle.dump(V_rough, open(f_rough, "wb"))
        return V_smooth.cuda(), V_rough.cuda()
