from torch.utils.data import Dataset
import numpy as np

class IPDataset(Dataset):
    def __init__(self, dataset, entity, mode):

        file = np.load(f"vectors/{dataset}/Clustering_s1234_{entity}_{mode}.npz", allow_pickle=True)
        self.x = file['x']
        self.y = file['y']
        self.n_samples = len(self.y)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]