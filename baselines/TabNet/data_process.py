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
    ip_path = '../../datasets/{}/ip.csv'.format(dataset)
    trace_path = '../../datasets/{}/last_traceroute.csv'.format(dataset)

    data_origin = pd.read_csv(data_path, encoding='gbk', low_memory=False)
    ip_origin = pd.read_csv(ip_path, encoding='gbk', low_memory=False)
    trace_origin = pd.read_csv(trace_path, encoding='gbk', low_memory=False)

    data = pd.concat([data_origin, ip_origin, trace_origin], axis=1)
    data.fillna({"isp": '0'}, inplace=True)

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

    lon_lat = data[['longitude', 'latitude']]
    lon_lat = np.array(lon_lat)

    return X, lon_lat