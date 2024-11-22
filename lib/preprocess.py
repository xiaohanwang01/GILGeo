# Load data and IP clustering

import math
import random
import pandas as pd
import numpy as np
import argparse
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.cluster import KMeans

from .utils import MaxMinScaler


def split_dataset(dataset):
    data_path = "../datasets/{}/data.csv".format(dataset)
    lat_lon = pd.read_csv(data_path, usecols=['latitude', 'longitude'], low_memory=False)

    labels = KMeans(n_clusters=2, random_state=0).fit(lat_lon).labels_
    indices1 = np.where(labels == 0)[0]
    indices2 = np.where(labels == 1)[0]
    if len(indices1) > len(indices2):
        train_idx = indices1
        test_idx = indices2
    else:
        train_idx = indices2
        test_idx = indices1
    return list(train_idx), list(test_idx)


def get_num(dataset):
    ip_path = './datasets/{}/ip.csv'.format(dataset)
    df = pd.read_csv(ip_path, encoding='gbk', low_memory=False)
    return len(df)


def get_XY(dataset):
    data_path = "./datasets/{}/data.csv".format(dataset)
    ip_path = './datasets/{}/ip.csv'.format(dataset)
    trace_path = './datasets/{}/last_traceroute.csv'.format(dataset)

    data_origin = pd.read_csv(data_path, encoding='gbk', low_memory=False)
    ip_origin = pd.read_csv(ip_path, encoding='gbk', low_memory=False)
    trace_origin = pd.read_csv(trace_path, encoding='gbk', low_memory=False)

    data = pd.concat([data_origin, ip_origin, trace_origin], axis=1)
    data.fillna({"isp": '0'}, inplace=True)

    # labels
    Y = data[['longitude', 'latitude']]
    Y = np.array(Y)

    # features
    if dataset == "Shanghai":  # Shanghai，27+8+16, 共51维，其中8+16=24维为traceroute相关measurment
        # classification features
        X_class = data[['orgname', 'asname', 'address', 'isp']]
        scaler = preprocessing.OneHotEncoder(sparse=False)
        X_class = scaler.fit_transform(X_class)

        X_class1 = data['isp']
        X_class1 = preprocessing.LabelEncoder().fit_transform(X_class1)
        X_class1 = preprocessing.MinMaxScaler().fit_transform(np.array(X_class1).reshape((-1, 1)))

        X_2 = data[['ip_split1', 'ip_split2', 'ip_split3', 'ip_split4']]
        X_2 = preprocessing.MinMaxScaler().fit_transform(np.array(X_2))

        X_3 = data['asnumber']
        X_3 = preprocessing.LabelEncoder().fit_transform(X_3)
        X_3 = preprocessing.MinMaxScaler().fit_transform(np.array(X_3).reshape(-1, 1))

        X_4 = data[['aiwen_ping_delay_time', 'vp806_ping_delay_time', 'vp808_ping_delay_time', 'vp813_ping_delay_time']]
        delay_scaler = MaxMinScaler()
        delay_scaler.fit(X_4)
        X_4 = delay_scaler.transform(X_4)

        X_5 = data[['aiwen_tr_steps', 'vp806_tr_steps', 'vp808_tr_steps', 'vp813_tr_steps']]
        step_scaler = MaxMinScaler()
        step_scaler.fit(X_5)
        X_5 = step_scaler.transform(X_5)

        X_6 = data[
            ['aiwen_last1_delay', 'aiwen_last2_delay_total', 'aiwen_last3_delay_total', 'aiwen_last4_delay_total',
             'vp806_last1_delay', 'vp806_last2_delay_total', 'vp806_last3_delay_total', 'vp806_last4_delay_total',
             'vp808_last1_delay', 'vp808_last2_delay_total', 'vp808_last3_delay_total', 'vp808_last4_delay_total',
             'vp813_last1_delay', 'vp813_last2_delay_total', 'vp813_last3_delay_total', 'vp813_last4_delay_total']]
        X_6 = np.array(X_6)
        X_6[X_6 <= 0] = 0
        X_6 = preprocessing.MinMaxScaler().fit_transform(X_6)

        X = np.concatenate([X_class1, X_class, X_2, X_3, X_4, X_5, X_6], axis=1)
        # without isp
        # X = np.concatenate([X_class, X_2, X_3, X_4, X_5, X_6], axis=1)

    elif dataset == "New_York" or "Los_Angeles":  # New_York or Los_Angeles, 6+8+16, 共30维, 其中8+16=24维为tracerout相关measurment
        X_class = data['isp']
        X_class = preprocessing.LabelEncoder().fit_transform(X_class)
        X_class = preprocessing.MinMaxScaler().fit_transform(np.array(X_class).reshape((-1, 1)))

        X_2 = data[['ip_split1', 'ip_split2', 'ip_split3', 'ip_split4']]
        X_2 = preprocessing.MinMaxScaler().fit_transform(np.array(X_2))

        X_3 = data['as_mult_info']
        X_3 = preprocessing.LabelEncoder().fit_transform(X_3)
        X_3 = preprocessing.MinMaxScaler().fit_transform(np.array(X_3).reshape(-1, 1))

        X_4 = data[['vp900_ping_delay_time', 'vp901_ping_delay_time', 'vp902_ping_delay_time', 'vp903_ping_delay_time']]
        delay_scaler = MaxMinScaler()
        delay_scaler.fit(X_4)
        X_4 = delay_scaler.transform(X_4)

        X_5 = data[['vp900_tr_steps', 'vp901_tr_steps', 'vp902_tr_steps', 'vp903_tr_steps']]
        step_scaler = MaxMinScaler()
        step_scaler.fit(X_5)
        X_5 = step_scaler.transform(X_5)

        X_6 = data[
            ['vp900_last1_delay', 'vp900_last2_delay_total', 'vp900_last3_delay_total', 'vp900_last4_delay_total',
             'vp901_last1_delay', 'vp901_last2_delay_total', 'vp901_last3_delay_total', 'vp901_last4_delay_total',
             'vp902_last1_delay', 'vp902_last2_delay_total', 'vp902_last3_delay_total', 'vp902_last4_delay_total',
             'vp903_last1_delay', 'vp903_last2_delay_total', 'vp903_last3_delay_total', 'vp903_last4_delay_total']]
        X_6 = np.array(X_6)
        X_6[X_6 <= 0] = 0
        X_6 = preprocessing.MinMaxScaler().fit_transform(X_6)

        X = np.concatenate([X_2, X_class, X_3, X_4, X_5, X_6], axis=1)
        # without isp
        # X = np.concatenate([X_2, X_3, X_4, X_5, X_6], axis=1)

    return X, Y, np.array(trace_origin)


def find_all_routers(row):
    last_router_idx = list(range(0, 32, 8))
    last_delay_idx = list(range(1, 32, 8))
    routers = row[last_router_idx]
    delays = row[last_delay_idx]
    delays[delays <= 0] = math.inf

    return routers, delays


def find_nearest_router(row):
    last_router_idx = list(range(0, 32, 8))
    last_delay_idx = list(range(1, 32, 8))
    routers = row[last_router_idx]
    delays = row[last_delay_idx]
    delays[delays <= 0] = math.inf
    nearest_idx = np.argmin(delays)

    return routers[nearest_idx], delays[nearest_idx]


def find_near2_router(row):

    last_router_idx = list(range(0, 32, 8))
    last_delay_idx = list(range(1, 32, 8))
    routers = row[last_router_idx]
    delays = row[last_delay_idx]
    delays[delays <= 0] = math.inf
    sort_idx = np.argsort(delays)

    return routers[sort_idx[:2]], delays[sort_idx[:2]]


def get_idx(idx, seed, train_test_ratio, lm_ratio):
    num = len(idx)
    random.seed(seed)
    random.shuffle(idx)
    lm_train_num = int(num * train_test_ratio * lm_ratio)
    tg_train_num = int(num * train_test_ratio * (1 - lm_ratio))

    lm_train_idx, tg_train_idx, tg_test_idx = idx[:lm_train_num], \
                                              idx[lm_train_num:tg_train_num + lm_train_num], \
                                              idx[lm_train_num + tg_train_num:]
    return lm_train_idx, tg_train_idx, lm_train_idx + tg_train_idx, tg_test_idx

def get_test_idx(idx, seed, lm_ratio):
    num = len(idx)
    random.seed(seed)
    random.shuffle(idx)
    lm_num = int(num * lm_ratio)

    lm_idx, tg_idx = idx[:lm_num], idx[lm_num:]
    return lm_idx, tg_idx

def get_graph(dataset, lm_idx, tg_idx, seed, mode, generalization_test=False):

    X, Y, T = get_XY(dataset)  # preprocess whole dataset

    last_hop = np.array(list(map(find_nearest_router, T)), dtype=object)  # [(ip, time delay),...]
    last_router = last_hop[:, 0]

    last_hops = np.array(list(map(find_near2_router, T)), dtype=object)
    last_routers = last_hops[:, 0]

    data_leq10 = []
    data_gt10 = {}

    for tg_id in tqdm(tg_idx):

        router = last_router[tg_id]
        if(router == '-1'):
            continue

        neighbors = []
        for lm_id in lm_idx:
            if last_router[lm_id] == router:
                neighbors.append(lm_id)

        if len(neighbors) == 0:
            continue

        if 0 < len(neighbors) <= 10:  

            leq_neighbors = set()
            tg_last_routers = set(last_routers[tg_id])
            tg_last_routers.discard('-1')
            for lm_id in lm_idx:
                lm_last_routers = set(last_routers[lm_id])
                lm_last_routers.discard('-1')
                if tg_last_routers & lm_last_routers: 
                    leq_neighbors.add(lm_id)
            leq_neighbors = np.array(list(leq_neighbors))

            data = {
                'lm_X': X[leq_neighbors],
                'lm_Y': Y[leq_neighbors],
                'tg_X': np.expand_dims(X[tg_id], axis=0),
                'tg_Y': np.expand_dims(Y[tg_id], axis=0)
            }
            data_leq10.append(data)

        if 10 < len(neighbors):
            if router not in data_gt10.keys():
                data = {
                    'lm_X': X[neighbors],
                    'lm_Y': Y[neighbors],
                    'tg_X': np.expand_dims(X[tg_id], axis=0),
                    'tg_Y': np.expand_dims(Y[tg_id], axis=0)
                }
                data_gt10[router] = data
            else:
                data_gt10[router]['tg_X'] = np.append(data_gt10[router]['tg_X'], np.expand_dims(X[tg_id], axis=0), axis=0)
                data_gt10[router]['tg_Y'] = np.append(data_gt10[router]['tg_Y'], np.expand_dims(Y[tg_id], axis=0), axis=0)
    
    data_gt10 = list(data_gt10.values())
    data = data_leq10 + data_gt10
    if generalization_test:
        np.savez("datasets/{}/generalization/Clustering_s{}_{}.npz".format(dataset, seed, mode), data=data)
    else:
        np.savez("datasets/{}/contrast/Clustering_s{}_{}.npz".format(dataset, seed, mode), data=data)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Shanghai', choices=["Shanghai", "New_York", "Los_Angeles"],
                        help='which dataset to use')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='landmark ratio')
    parser.add_argument('--lm_ratio', type=float, default=0.7, help='landmark ratio')
    parser.add_argument('--seed', type=int, default=1234)
    
    
    opt = parser.parse_args()
    print("Dataset: ", opt.dataset)
    seed = opt.seed
    train_ratio = opt.train_ratio
    lm_ratio = opt.lm_ratio
    city = opt.dataset

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