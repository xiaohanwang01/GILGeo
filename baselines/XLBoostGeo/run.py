import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import ipdb
from tqdm import tqdm

from data_process import split_dataset, get_test_idx, get_data


def get_mselist(y, y_pred):
    mse = (((y - y_pred) * 100) ** 2).sum(axis=1)
    return mse


def get_r_hops(trace):
    r_hop = {}
    for route in trace:
        for i, r in enumerate(route[:-1]):
            if len(r) != 0:
                key = list(r.keys())[0]
                hop = len(route)-i-1
                if key not in r_hop.keys() or r_hop[key] > hop:
                    r_hop[key] = hop
    return r_hop


def get_topology_score(tg_r_hop, lm_r_hop):
    min_hop = 30
    tg_set = set(tg_r_hop.keys())
    lm_set = set(lm_r_hop.keys())
    intersection_set = tg_set & lm_set
    if len(intersection_set) == 0:
        return min_hop
    for r in intersection_set:
        hop = tg_r_hop[r] + lm_r_hop[r]
        if hop < min_hop:
            min_hop = hop
    return hop


if __name__ == '__main__':
    datasets = ["New_York", "Los_Angeles", "Shanghai"]
    for dataset in datasets:
        lm_ratio = 0.7
        seed = 1234
        print("Dataset: ", dataset)
        _, test_idx = split_dataset(dataset)
        test_lm_idx, test_tg_idx = get_test_idx(test_idx, seed, lm_ratio) 
        delay, traces, lon_lat = get_data(dataset)
        
        sim = cosine_similarity(delay[test_tg_idx], delay[test_lm_idx])
        sim_distance = sim / (sim.sum(axis=-1, keepdims=True)+1e-12)

        sim_topology = np.ones_like(sim)

        tg_r_hops = []
        lm_r_hops = []
        
        for tg_idx in tqdm(range(sim_topology.shape[0])):
            tg_r_hops.append(get_r_hops(traces[test_tg_idx][tg_idx]))
        for lm_idx in tqdm(range(sim_topology.shape[1])):
            lm_r_hops.append(get_r_hops(traces[test_lm_idx][lm_idx]))
        

        for tg_idx in tqdm(range(sim_topology.shape[0])):
            for lm_idx in range(sim_topology.shape[1]):
                sim_topology[tg_idx][lm_idx] = get_topology_score(tg_r_hops[tg_idx], lm_r_hops[lm_idx])

        sim_topology = 1 - np.exp(sim_topology) / np.exp(sim_topology).sum(axis=1, keepdims=True)

        sim = sim_distance + sim_topology
        idx = np.argmax(sim, axis=-1)
        preds = lon_lat[test_lm_idx][idx]
        y = lon_lat[test_tg_idx]

        mse = get_mselist(y, preds)
        distance = np.sqrt(mse)
        sorted_distance = np.sort(distance)

        print(np.sqrt(mse.mean()))
        print(distance.mean())
        print(sorted_distance[int(len(sorted_distance)/2)])