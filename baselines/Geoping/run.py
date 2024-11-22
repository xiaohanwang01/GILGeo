import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_process import split_dataset, get_test_idx, get_data


def get_mselist(y, y_pred):
    mse = (((y - y_pred) * 100) ** 2).sum(axis=1)
    return mse


if __name__ == '__main__':
    datasets = ["New_York", "Los_Angeles", "Shanghai"]
    for dataset in datasets:
        lm_ratio = 0.7
        seed = 1234
        print("Dataset: ", dataset)
        train_idx, test_idx = split_dataset(dataset)
        test_lm_idx, test_tg_idx = get_test_idx(test_idx, seed, lm_ratio) 
        delay, _, lon_lat = get_data(dataset)
        sim = cosine_similarity(delay[test_tg_idx], delay[test_lm_idx])
        index = sim.argmax(axis=1)
        preds = lon_lat[test_lm_idx][index]

        mse = get_mselist(lon_lat[test_tg_idx], preds)
        distance = np.sqrt(mse)
        sorted_distance = np.sort(distance)

        print(np.sqrt(mse.mean()))
        print(distance.mean())
        print(sorted_distance[int(len(sorted_distance)/2)])