import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import os

from tqdm import tqdm


def cluster(dataset, mode):
    data = np.load(f"vectors/{dataset}/Clustering_s1234_lm_{mode}.npz", allow_pickle=True)
    x = data['x']
    y = data['y']
    x_scores_max = 0
    y_scores_max = 0
    for k in tqdm(range(2, 7)):
        x_kmeans = KMeans(n_clusters=k).fit(x)
        x_score = silhouette_score(x,x_kmeans.labels_,metric='euclidean')
        if x_score >= x_scores_max:
            x_scores_max = x_score
            x_k = k
            x_model = x_kmeans
        y_kmeans = KMeans(n_clusters=k).fit(y)
        y_score = silhouette_score(y,y_kmeans.labels_,metric='euclidean')
        if y_score >= y_scores_max:
            y_scores_max = y_score
            y_k = k
            y_model = y_kmeans
    
    print(f'x_k: {x_k}, y_k: {y_k}')
    merge_idx = []
    for i in range(x_k):
        x_idx = set(np.where(x_model.labels_==i)[0])
        for j in range(y_k):
            y_idx = set(np.where(y_model.labels_==j)[0])
            merge_idx.append(list(x_idx.intersection(y_idx)))

    with open(f'clusters/{dataset}_{mode}_clusters.pkl', 'wb') as f:
        pickle.dump(merge_idx, f)
    print(f'{dataset} {mode} set clustering has been completed')
    
    
if __name__ == '__main__':
    datasets = ["New_York", "Los_Angeles", "Shanghai"]
    modes = ['train', 'valid', 'test']
    for dataset in datasets:
        data_path = f'clusters/{dataset}'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        for mode in modes:
            cluster(dataset, mode)