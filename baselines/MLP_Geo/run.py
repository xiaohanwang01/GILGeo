import torch
import numpy as np
import argparse
import pickle, os, csv
from sklearn.metrics.pairwise import cosine_similarity

from model import MLPGeo
from utils import get_mselist

def load_args():
    parser = argparse.ArgumentParser('1')
    # parameters of initializing
    parser.add_argument('--seed', type=int, default=0, help='manual seed')
    parser.add_argument('--dataset', type=str, default='New_York', choices=["New_York", "Los_Angeles", "Shanghai"])
    parser.add_argument('--gpu', type=int, default=1, help='select gpu')
    # parameters of training
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--harved_epoch', type=int, default=10) 
    parser.add_argument('--early_stop_epoch', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=500)
    # parameters of model
    parser.add_argument('--input_size', type=int, default=1973)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--output_size', type=int, default=2)
    opt = parser.parse_args()

    return opt


def run():
    opt = load_args()

    if isinstance(opt.seed, int):
        print("Random Seed: ", opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)

    dataset = opt.dataset
    print("Dataset: ", dataset)
    train_data_x = torch.tensor(np.load(f'vectors/{dataset}/Clustering_s1234_lm_train.npz')['x']).cuda(opt.gpu).float()
    valid_data_x = torch.tensor(np.load(f'vectors/{dataset}/Clustering_s1234_tg_train.npz')['x']).cuda(opt.gpu).float()
    test_data_x =  torch.tensor(np.load(f'vectors/{dataset}/Clustering_s1234_tg_test.npz')['x']).cuda(opt.gpu).float()
    train_data_y = torch.tensor(np.load(f'vectors/{dataset}/Clustering_s1234_lm_train.npz')['y']).cuda(opt.gpu).float()
    valid_data_y = torch.tensor(np.load(f'vectors/{dataset}/Clustering_s1234_tg_train.npz')['y']).cuda(opt.gpu).float()
    test_data_y =  torch.tensor(np.load(f'vectors/{dataset}/Clustering_s1234_tg_test.npz')['y']).cuda(opt.gpu).float()

    with open(f'clusters/{opt.dataset}_train_clusters.pkl', 'rb') as f:
        cluster = pickle.load(f)
    
    model = MLPGeo(opt.input_size, opt.hidden_size, opt.output_size, amount=len(cluster))
    model.cuda(opt.gpu)
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    log_dir = f"log"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = os.path.join(log_dir, f'{opt.dataset}.csv')
    best_metric = {'epoch':0, 'id-mse':0, 'id-rmse':0, 'id-mae':0, 'id-median':0, 'ood-mse':0, 'ood-rmse':0, 'ood-mae':0, 'ood-median':0}
    header = ['seed'] + list(best_metric.keys())
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)

    losses = [np.inf]
    no_better_epoch = 0
    early_stop_epoch = 0
    
    cluster_center = torch.concat([train_data_x[cluster[i]].mean(dim=0).unsqueeze(dim=0) for i in range(len(cluster))], dim=0).cuda(opt.gpu)
    for epoch in range(1, opt.epochs+1):
        print("epoch {}.    ".format(epoch))
        total_mse, total_mae, train_num = 0, 0, 0
        model.train()
        
        loss = 0
        for i, ids in enumerate(cluster):
            x = train_data_x[ids]
            y = train_data_y[ids]

            preds = model(x, i)
            mse = get_mselist(y, preds)
            distance = torch.sqrt(mse)

            loss += mse.sum()

            total_mse += mse.sum()
            total_mae += distance.sum()
            train_num += len(y)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_mse /= train_num
        total_mae /= train_num

        print("train: mse loss: {:.4f} mae: {:.4f}".format(total_mse, total_mae))

        model.eval()
        with torch.no_grad():
            # valid
            v_mse, v_mae, v_num = 0, 0, 0
            v_distance = []

            sim = cosine_similarity(valid_data_x.cpu(), cluster_center.cpu())
            ids = torch.tensor(np.argmax(sim, axis=-1))
            for i in range(ids.max()):
                idx = torch.where(ids == i)[0]
                x = valid_data_x[idx]
                y = valid_data_y[idx]

                preds = model(x, i)
                mse = get_mselist(y, preds)
                distance = torch.sqrt(mse)

                for j in range(len(distance.cpu().detach().numpy())):
                    v_distance.append(distance.cpu().detach().numpy()[j])

                v_mse += mse.sum()
                v_mae += distance.sum()
                v_num += len(y)

            v_mse /= v_num
            v_mae /= v_num
            v_distance = sorted(v_distance)
            v_median = v_distance[int(len(v_distance) / 2)]
            print("Valid: mse: {:.4f}  mae: {:.4f}  median {:.4f}".format(v_mse, v_mae, v_median))


            # test
            t_mse, t_mae, t_num = 0, 0, 0
            t_distance = []
            sim = cosine_similarity(test_data_x.cpu(), cluster_center.cpu())
            ids = torch.tensor(np.argmax(sim, axis=-1))
            for i in range(ids.max()):
                idx = torch.where(ids == i)[0]
                x = test_data_x[idx]
                y = test_data_y[idx]
           
                preds = model(x, i)         
                mse = get_mselist(y, preds)
                distance = torch.sqrt(mse)

                for j in range(len(distance.cpu().detach().numpy())):
                    t_distance.append(distance.cpu().detach().numpy()[j])
                
                t_mse += mse.sum()
                t_mae += distance.sum()
                t_num += len(y)

            t_mse /= t_num
            t_mae /= t_num
            t_distance = sorted(t_distance)
            t_median = t_distance[int(len(t_distance) / 2)]
            print("test: mse: {:.4f}  mae: {:.4f}  median {:.4f}".format(t_mse, t_mae, t_median))

            batch_mae = v_mae.cpu().numpy()
            if batch_mae < np.min(losses):
                best_metric['epoch'] = epoch
                best_metric['id-mse'] = v_mse.item()
                best_metric['id-rmse'] = torch.sqrt(v_mse).item()
                best_metric['id-mae'] = v_mae.item()
                best_metric['id-median'] = v_median
                best_metric['ood-mse'] = t_mse.item()
                best_metric['ood-rmse'] = torch.sqrt(t_mse).item()
                best_metric['ood-mae'] = t_mae.item()
                best_metric['ood-median'] = t_median        
                no_better_epoch = 0 
                early_stop_epoch = 0
                print("Better MAE in epoch {}: {:.4f}".format(epoch, batch_mae))
            else:
                no_better_epoch = no_better_epoch + 1
                early_stop_epoch = early_stop_epoch + 1

            losses.append(batch_mae)

        # halve the learning rate
        if no_better_epoch % opt.harved_epoch == 0 and no_better_epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 2
                print("learning rate changes to {}!\n".format(param_group['lr']))
            no_better_epoch = 0

        if early_stop_epoch == opt.early_stop_epoch:
            break

    print(best_metric)
    metric = [opt.seed] + list(best_metric.values())
    with open(log_file, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(metric)

if __name__ == '__main__':
    run()