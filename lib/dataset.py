import torch
from torch_geometric.data import InMemoryDataset, Data
import os
import os.path as osp 
import numpy as np

from .preprocess import * 


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, city, mode, generalization_test=False, norm_x=False, train_ratio=0.8, lm_ratio=0.7, seed=1234):
        self.city = city
        self.generalization_test = generalization_test
        self.norm_x = norm_x
        self.train_ratio = train_ratio
        self.lm_ratio = lm_ratio
        self.seed = seed
        
        if(generalization_test):
            root = osp.join(root, city, 'generalization')
        else:
            root = osp.join(root, city, 'contrast')
        if not osp.exists(root):
            os.makedirs(root)

        super().__init__(root)

        if generalization_test:
            if mode == 'train':
                path = self.processed_paths[0]
            elif mode == 'valid':
                path = self.processed_paths[1]
            elif mode == 'test':
                path = self.processed_paths[2]
            else:
                raise ValueError(f"Split '{mode}' found, but expected either "
                                 f"'train', 'val', or 'test'")
        else:
            if mode == 'train':
                path = self.processed_paths[0]
            elif mode == 'test':
                path = self.processed_paths[1]
            else:
                raise ValueError(f"Split '{mode}' found, but expected either "
                             f"'train', 'val', or 'test'")
        
        self.load(path)


    @property
    def raw_dir(self) -> str:
        return self.root
    
    @property
    def processed_dir(self) -> str:
        return self.root
    
    @property
    def raw_file_names(self):
        if self.generalization_test:  
            return [f'Clustering_s{self.seed}_train.npz', f'Clustering_s{self.seed}_valid.npz', f'Clustering_s{self.seed}_test.npz']
        else:
            return [f'Clustering_s{self.seed}_train.npz', f'Clustering_s{self.seed}_test.npz']
    
    @property
    def processed_file_names(self):
        if self.generalization_test:  
            if self.norm_x:
                return ['norm_x_train.pt', 'norm_x_valid.pt', 'norm_x_test.pt']
            else:
                return ['unnorm_x_train.pt', 'unnorm_x_valid.pt', 'unnorm_x_test.pt']
        else:
            if self.norm_x:
                return ['norm_x_train.pt', 'norm_x_test.pt']
            else:
                return ['unnorm_x_train.pt', 'unnorm_x_test.pt']
        
    def download(self):
        city = self.city
        generalization_test = self.generalization_test
        train_ratio = self.train_ratio
        lm_ratio = self.lm_ratio
        seed = self.seed

        print(f"dataset: {city}")

        if generalization_test:
        # generalization
            train_idx, test_idx = split_dataset(city)
            train_lm_idx, train_tg_idx, valid_lm_idx, valid_tg_idx = get_idx(train_idx, seed, train_ratio, lm_ratio)  # split train and test
            test_lm_idx, test_tg_idx = get_test_idx(test_idx, seed, lm_ratio)

            print(f"loading generalization experiment data...")
            print("loading train set...")
            get_graph(city, train_lm_idx, train_tg_idx, seed, mode="train", generalization_test=True)
            print("train set loaded.")

            print("loading valid set...")
            get_graph(city, valid_lm_idx, valid_tg_idx, seed, mode="valid", generalization_test=True)
            print("valid set loaded.")

            print("loading test set...")
            get_graph(city, test_lm_idx, test_tg_idx, seed, mode="test", generalization_test=True)
            print("test set loaded.")

            print("finish!")

        # contrast
        else:
            idx = list(range(get_num(city)))
            train_lm_idx, train_tg_idx, test_lm_idx, test_tg_idx = get_idx(idx, seed, train_ratio, lm_ratio)  # split train and test
            print("loading train set...")
            get_graph(city, train_lm_idx, train_tg_idx, seed, mode="train")
            print("train set loaded.")

            print("loading test set...")
            get_graph(city, test_lm_idx, test_tg_idx, seed, mode="test")
            print("test set loaded.")

            print("finish!")

    
    def process(self):
        for i in range(len(self.processed_file_names)):
            data = np.load(self.raw_paths[i], allow_pickle=True)["data"]
            data = self.graph_normal(data, norm_x=self.norm_x)
            data_list = []
            for g in data:
                N1 = len(g['lm_Y'])
                N2 = len(g['tg_Y'])
                edge_index = self.get_adj(g)
                # lm_feature = np.concatenate((g['lm_X'], g['lm_delay'][:, None], g['lm_Y']), axis=1)
                # tg_feature = np.concatenate((g['tg_X'], g['tg_delay'][:, None], np.zeros_like(g['tg_Y'])), axis=1)
                lm_feature = np.concatenate((g['lm_X'], g['lm_Y']), axis=1)
                tg_feature = np.concatenate((g['tg_X'], np.zeros_like(g['tg_Y'])), axis=1)
                x = torch.FloatTensor(np.concatenate((lm_feature, tg_feature), axis=0))
                y = torch.FloatTensor(np.concatenate((g['lm_Y'], g['tg_Y']), axis=0))
                tg_mask = torch.cat((torch.zeros(N1), torch.ones(N2)))
                y_max = torch.from_numpy(g['y_max']).unsqueeze(0).repeat(N1+N2, 1)
                y_min = torch.from_numpy(g['y_min']).unsqueeze(0).repeat(N1+N2, 1)
                # y_range = torch.from_numpy(g['y_max'] - g['y_min']).unsqueeze(0).repeat(N1+N2, 1)
                data_list.append(Data(x=x, edge_index=edge_index, y=y, tg_mask=tg_mask, y_max=y_max, y_min=y_min))
                # data_list.append(Data(x=x, edge_index=edge_index, y=y, tg_mask=tg_mask))
            self.save(data_list, self.processed_paths[i])

    def get_adj(self, graph):
        N1 = len(graph['lm_Y'])
        N2 = len(graph['tg_Y'])
        # target = [i for i in range(N1) for j in range(N2)]  + [i+N1 for i in range(N2) for j in range(N1)]
        # source = [j+N1 for i in range(N1) for j in range(N2)] + [j for i in range(N2) for j in range(N1)]
        target = [i+N1 for i in range(N2) for j in range(N1)]
        source = [j for i in range(N2) for j in range(N1)]
        edge_index = [source, target]
        # delay_matrix = np.concatenate((graph['lm_delay'], graph['tg_delay']))
        # delay1 = delay_matrix[source]
        # delay2 = delay_matrix[target]
        # edge_attr = np.concatenate((delay1[:, None], delay2[:, None]), axis=1)

        # return torch.LongTensor(edge_index), torch.FloatTensor(edge_attr)
        return torch.LongTensor(edge_index)
    
    
    def graph_normal(self, graphs, norm_x):
        for g in graphs:
            if norm_x:
                X = np.concatenate((g["lm_X"], g["tg_X"]), axis=0)  
                g["lm_X"] = (g["lm_X"] - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
                g["tg_X"] = (g["tg_X"] - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)

            Y = np.concatenate((g["lm_Y"], g["tg_Y"]), axis=0)
            g["lm_Y"] = (g["lm_Y"] - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["tg_Y"] = (g["tg_Y"] - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["y_max"], g["y_min"] = Y.max(axis=0), Y.min(axis=0)

        return graphs   


class HyperGraphDataset(InMemoryDataset):
    def __init__(self, root, city, mode, generalization_test=False, norm_x=False, train_ratio=0.8, lm_ratio=0.7, seed=1234):
        self.city = city
        self.generalization_test = generalization_test
        self.norm_x = norm_x
        self.train_ratio = train_ratio
        self.lm_ratio = lm_ratio
        self.seed = seed
        
        if(generalization_test):
            root = osp.join(root, city, 'generalization')
        else:
            root = osp.join(root, city, 'contrast')
        if not osp.exists(root):
            os.makedirs(root)

        super().__init__(root)

        if generalization_test:
            if mode == 'train':
                path = self.processed_paths[0]
            elif mode == 'valid':
                path = self.processed_paths[1]
            elif mode == 'test':
                path = self.processed_paths[2]
            else:
                raise ValueError(f"Split '{mode}' found, but expected either "
                                 f"'train', 'val', or 'test'")
        else:
            if mode == 'train':
                path = self.processed_paths[0]
            elif mode == 'test':
                path = self.processed_paths[1]
            else:
                raise ValueError(f"Split '{mode}' found, but expected either "
                             f"'train', 'val', or 'test'")
        
        self.load(path)


    @property
    def raw_dir(self) -> str:
        return self.root
    
    @property
    def processed_dir(self) -> str:
        return self.root
    
    @property
    def raw_file_names(self):
        if self.generalization_test:  
            return [f'Clustering_s{self.seed}_train.npz', f'Clustering_s{self.seed}_valid.npz', f'Clustering_s{self.seed}_test.npz']
        else:
            return [f'Clustering_s{self.seed}_train.npz', f'Clustering_s{self.seed}_test.npz']
    
    @property
    def processed_file_names(self):
        if self.generalization_test:  
            if self.norm_x:
                return ['norm_x_train.pt', 'norm_x_valid.pt', 'norm_x_test.pt']
            else:
                return ['unnorm_x_hypergraph_train.pt', 'unnorm_x_hypergraph_valid.pt', 'unnorm_x_hypergraph_test.pt']
        else:
            if self.norm_x:
                return ['norm_x_train.pt', 'norm_x_test.pt']
            else:
                return ['unnorm_x_train.pt', 'unnorm_x_test.pt']
        
    def download(self):
        pass

    def process(self):
        for i in range(len(self.processed_file_names)):
            data = np.load(self.raw_paths[i], allow_pickle=True)["data"]
            data = self.graph_normal(data, norm_x=self.norm_x)
            data_list = []
            for g in data:
                N1 = len(g['lm_Y'])
                N2 = len(g['tg_Y'])
                hyperedge_index = self.get_hadj(g)
                # lm_feature = np.concatenate((g['lm_X'], g['lm_delay'][:, None], g['lm_Y']), axis=1)
                # tg_feature = np.concatenate((g['tg_X'], g['tg_delay'][:, None], np.zeros_like(g['tg_Y'])), axis=1)
                lm_feature = np.concatenate((g['lm_X'], g['lm_Y']), axis=1)
                tg_feature = np.concatenate((g['tg_X'], np.zeros_like(g['tg_Y'])), axis=1)
                x = torch.FloatTensor(np.concatenate((lm_feature, tg_feature), axis=0))
                y = torch.FloatTensor(np.concatenate((g['lm_Y'], g['tg_Y']), axis=0))
                tg_mask = torch.cat((torch.zeros(N1), torch.ones(N2)))
                y_max = torch.from_numpy(g['y_max']).unsqueeze(0).repeat(N1+N2, 1)
                y_min = torch.from_numpy(g['y_min']).unsqueeze(0).repeat(N1+N2, 1)
                # y_range = torch.from_numpy(g['y_max'] - g['y_min']).unsqueeze(0).repeat(N1+N2, 1)
                data_list.append(Data(x=x, hyperedge_index=hyperedge_index, y=y, tg_mask=tg_mask, y_max=y_max, y_min=y_min))
                # data_list.append(Data(x=x, edge_index=edge_index, y=y, tg_mask=tg_mask))
            self.save(data_list, self.processed_paths[i])

    def get_hadj(self, graph):
        lm_x = graph['lm_X']
        tg_x = graph['tg_X']
        num_column = tg_x.shape[1]
        x = np.concatenate((lm_x, tg_x), axis=0)
        hyperedge_index = []
        for col_id in range(num_column):
            x_col = x[:, col_id]
            center = np.argmin(np.abs(x_col-x_col.mean()))
            dis =(x_col[center]-x_col) ** 2
            k = int(len(dis) / 2)
            idx = np.argsort(dis)[:k]
            for id in idx:
                hyperedge_index.append([id, col_id])
        hyperedge_index = torch.LongTensor(hyperedge_index).T

        return hyperedge_index
    
    
    def graph_normal(self, graphs, norm_x):
        for g in graphs:
            if norm_x:
                X = np.concatenate((g["lm_X"], g["tg_X"]), axis=0)  
                g["lm_X"] = (g["lm_X"] - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
                g["tg_X"] = (g["tg_X"] - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)

            Y = np.concatenate((g["lm_Y"], g["tg_Y"]), axis=0)
            g["lm_Y"] = (g["lm_Y"] - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["tg_Y"] = (g["tg_Y"] - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["y_max"], g["y_min"] = Y.max(axis=0), Y.min(axis=0)
        
        return graphs