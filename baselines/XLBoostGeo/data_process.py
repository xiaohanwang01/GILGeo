# Load data and IP clustering

import random

import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans

from tqdm import tqdm
import ipdb

class MaxMinScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def fit(self, data):
        self.max = data.max()
        self.min = data.min()

    def transform(self, data):
        max = self.max
        min = self.min
        return (data - min) / (max - min + 1e-12)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min

def split_dataset(dataset):
    data_path = "../../datasets/{}/data.csv".format(dataset)
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

def get_train_idx(idx, seed, train_test_ratio, lm_ratio):
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
    lm_test_num = int(num * lm_ratio)
    lm_test_idx, tg_test_idx = idx[:lm_test_num], idx[lm_test_num:]
    return lm_test_idx, tg_test_idx


def get_data(dataset):
    data_path = "../../datasets/{}/data.csv".format(dataset)
    data = pd.read_csv(data_path, encoding='gbk', low_memory=False)

    # features
    if dataset == "Shanghai":  
        delay = data[['aiwen_ping_delay_time', 'vp806_ping_delay_time', 'vp808_ping_delay_time', 'vp813_ping_delay_time']]
        delay = np.array(delay)
        delay_scaler = MaxMinScaler()
        delay_scaler.fit(delay)
        delay = delay_scaler.transform(delay)

        traces = data[['aiwen_trace', 'vp806_trace', 'vp808_trace', 'vp813_trace']]
        traces = np.vectorize(eval)(traces.values)

    elif dataset == "New_York" or "Los_Angeles": 
        delay = data[['vp900_ping_delay_time', 'vp901_ping_delay_time', 'vp902_ping_delay_time', 'vp903_ping_delay_time']]
        delay = np.array(delay)
        delay_scaler = MaxMinScaler()
        delay_scaler.fit(delay)
        delay = delay_scaler.transform(delay)

        traces = data[['vp900_trace', 'vp901_trace', 'vp902_trace', 'vp903_trace']]
        # ipdb.set_trace()
        traces = np.vectorize(eval)(traces.values)

    lon_lat = data[['longitude', 'latitude']]
    lon_lat = np.array(lon_lat)

    return delay, traces, lon_lat