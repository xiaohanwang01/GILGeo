import numpy as np
import os
from tqdm import tqdm
from data_process import split_dataset, get_train_idx, get_test_idx, get_data

def encode_path(lm_traces, beta=30, router_set=None):
    traces = lm_traces.T
    num_vp = traces.shape[0]
    num_lm = traces.shape[1]
    router_path = [[{} for _ in range(num_lm)] for _ in range(num_vp)]
    if router_set is None:
        router_collect = True
        router_set = [{} for _ in range(num_vp)]
    else:
        router_collect = False
    for vp_id in range(num_vp):
        for lm_id in range(num_lm):
            trace = traces[vp_id][lm_id]
            for idx in range(len(trace)-1):
                if len(trace[idx]) != 0:
                    key = list(trace[idx].keys())[0]
                    router_path[vp_id][lm_id][key] = len(trace)-1-idx
                    if router_collect:
                        if key not in router_set[vp_id].keys():
                            router_set[vp_id][key] = 1
                        else:
                            router_set[vp_id][key] += 1
    if router_collect:
        for vp_id in range(num_vp):
            router_set[vp_id] = [k for k,v in router_set[vp_id].items() if v>=5]

    vectors = [[] for _ in range(num_lm)]
    for lm_id in tqdm(range(num_lm)):
        for vp_id in range(num_vp):
            for r in router_set[vp_id]:
                if r in router_path[vp_id][lm_id].keys():
                    value = router_path[vp_id][lm_id][r]
                else:
                    value = beta
                vectors[lm_id].append(value)
    vectors = np.array(vectors)

    if router_collect:
        return vectors, router_set
    else:
        return vectors


def construct_vector(dataset):
    data_path = f'vectors/{dataset}'
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    train_test_ratio = 0.8
    lm_ratio = 0.7
    seed = 1234
    print("Dataset: ", dataset)
    train_idx, test_idx = split_dataset(dataset)
    train_lm_idx, train_tg_idx, valid_lm_idx, valid_tg_idx = get_train_idx(train_idx, seed, train_test_ratio, lm_ratio)
    test_lm_idx, test_tg_idx = get_test_idx(test_idx, seed, lm_ratio) 
    delay, traces, lon_lat = get_data(dataset)
    
    train_lm_vectors, router_set = encode_path(traces[train_lm_idx])
    train_lm_x = np.concatenate((delay[train_lm_idx], train_lm_vectors), axis=1)
    train_lm_y = lon_lat[train_lm_idx]
    np.savez(f"vectors/{dataset}/Clustering_s{seed}_lm_train.npz", x=train_lm_x, y=train_lm_y) # 
    print(f"Vector construction of landmark's delay and path in train set have been completed!")

    train_tg_vectors = encode_path(traces[train_tg_idx], router_set=router_set)
    train_tg_x = np.concatenate((delay[train_tg_idx], train_tg_vectors), axis=1)
    train_tg_y = lon_lat[train_tg_idx]
    np.savez(f"vectors/{dataset}/Clustering_s{seed}_tg_train.npz", x=train_tg_x, y=train_tg_y) # 
    print(f"Vector construction of target's delay and path in train set have been completed!")

    valid_lm_vectors = encode_path(traces[valid_lm_idx], router_set=router_set)
    valid_lm_x = np.concatenate((delay[valid_lm_idx], valid_lm_vectors), axis=1)
    valid_lm_y = lon_lat[valid_lm_idx]
    np.savez(f"vectors/{dataset}/Clustering_s{seed}_lm_valid.npz", x=valid_lm_x, y=valid_lm_y)
    print(f"Vector construction of landmark's delay and path in valid set have been completed!")
    
    valid_tg_vectors = encode_path(traces[valid_tg_idx], router_set=router_set)
    valid_tg_x = np.concatenate((delay[valid_tg_idx], valid_tg_vectors), axis=1)
    valid_tg_y = lon_lat[valid_tg_idx]
    np.savez(f"vectors/{dataset}/Clustering_s{seed}_tg_valid.npz", x=valid_tg_x, y=valid_tg_y)
    print(f"Vector construction of target's delay and path in valid set have been completed!")
    
    test_lm_vectors = encode_path(traces[test_lm_idx], router_set=router_set)
    test_lm_x = np.concatenate((delay[test_lm_idx], test_lm_vectors), axis=1)
    test_lm_y = lon_lat[test_lm_idx]
    np.savez(f"vectors/{dataset}/Clustering_s{seed}_lm_test.npz", x=test_lm_x, y=test_lm_y)
    print(f"Vector construction of landmark's delay and path in test set have been completed!")

    test_tg_vectors = encode_path(traces[test_tg_idx], router_set=router_set)
    test_tg_x = np.concatenate((delay[test_tg_idx], test_tg_vectors), axis=1)
    test_tg_y = lon_lat[test_tg_idx]
    np.savez(f"vectors/{dataset}/Clustering_s{seed}_tg_test.npz", x=test_tg_x, y=test_tg_y) # 
    print(f"Vector construction of target's delay and path in test set have been completed!")



if __name__ == '__main__':
    datasets = ["New_York", "Los_Angeles", "Shanghai"]
    for dataset in datasets:
        construct_vector(dataset)