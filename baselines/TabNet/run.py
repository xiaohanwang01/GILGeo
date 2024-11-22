import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

import argparse
import os, csv

import ipdb
from tqdm import tqdm

from data_process import split_dataset, get_train_idx, get_test_idx, get_data


def get_mselist(y, y_pred):
    mse = (((y - y_pred) * 100) ** 2).sum(axis=1)
    return mse
    
def load_args():
    parser = argparse.ArgumentParser('1')
    # parameters of initializing
    parser.add_argument('--seed', type=int, default=0, help='manual seed')
    parser.add_argument('--dataset', type=str, default='New_York', choices=["New_York", "Los_Angeles", "Shanghai"])
    parser.add_argument('--lr', type=float, default=1e-2)
    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    opt = load_args()

    log_dir = f"log"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = os.path.join(log_dir, f'{opt.dataset}.csv')
    best_metric = {'ood-mse':0, 'ood-rmse':0, 'ood-mae':0, 'ood-median':0}
    header = ['seed'] + list(best_metric.keys())
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)

    train_test_ratio = 0.8
    lm_ratio = 0.7
    split_seed = 1234
    
    print("Dataset: ", opt.dataset)
    train_idx, test_idx = split_dataset(opt.dataset)
    train_lm_idx, train_tg_idx, _, _ = get_train_idx(train_idx, split_seed, train_test_ratio, lm_ratio)
    _, test_tg_idx = get_test_idx(test_idx, split_seed, lm_ratio) 
    x, lon_lat = get_data(opt.dataset)
    
    x_train = x[train_lm_idx]
    y_train = lon_lat[train_lm_idx]

    x_valid = x[train_tg_idx]
    y_valid = lon_lat[train_tg_idx]

    x_test = x[test_tg_idx]
    y_test = lon_lat[test_tg_idx]
    
    clf = TabNetRegressor(seed=opt.seed, optimizer_params={'lr':opt.lr}, device_name='cuda')  #TabNetRegressor()
    clf.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric=['mae']
    )
    preds = clf.predict(x_test)
    # preds = clf(x_test)
    # ipdb.set_trace()
    mse = get_mselist(y_test, preds)
    distance = np.sqrt(mse)
    sorted_distance = np.sort(distance)
    
    best_metric['ood-mse'] = mse.mean()
    best_metric['ood-rmse'] = np.sqrt(mse.mean())
    best_metric['ood-mae'] = distance.mean()
    best_metric['ood-median'] = sorted_distance[int(len(sorted_distance)/2)]   

    print(best_metric)
    metric = [opt.seed] + list(best_metric.values())
    with open(log_file, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(metric)