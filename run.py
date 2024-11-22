# -*- coding: utf-8 -*-
import torch.nn
import argparse, os, csv
import numpy as np
import random
import setproctitle

from lib.model import *
from lib.datasets import MyOwnDataset
from lib.utils import *
from torch_geometric.loader import DataLoader


def load_args():
    parser = argparse.ArgumentParser('1')
    # parameters of initializing
    parser.add_argument('--seed', type=int, default=0, help='manual seed')
    parser.add_argument('--model_name', type=str, default='GILGeo')
    parser.add_argument('--dataset', type=str, default='New_York', choices=["New_York", "Los_Angeles", "Shanghai"])
    parser.add_argument('--gpu', type=int, default=0, help='select gpu')
    parser.add_argument('--batch_size', type=int, default=4) # best value is 4
    parser.add_argument('--norm_x', action="store_true")
    # parameters of training
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--harved_epoch', type=int, default=5) 
    parser.add_argument('--early_stop_epoch', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--fea', type=float, default=1e-3)
    parser.add_argument('--edge', type=float, default=1)
    parser.add_argument('--fix_r', type=float, default=0.7)
    parser.add_argument('--l2', type=float, default=1e-2)
    # parameters of model
    parser.add_argument('--dim_in', type=int, default=32, choices=[32, 53], help="2 for latitude and longitude")
    opt = parser.parse_args()

    return opt


def print_args(opt):
    print("Learning rate: ", opt.lr)
    print("Dataset: ", opt.dataset)
    print("Model: ", opt.model_name)
    if isinstance(opt.seed, int): 
        print("Random Seed: ", opt.seed)
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
    torch.set_printoptions(threshold=float('inf'))
    print("Gpu:", opt.gpu)


if __name__ == '__main__':
    # Set the process title
    setproctitle.setproctitle("wxh_ip_gnn")

    opt = load_args()
    print_args(opt)

    '''load data'''
    train_data = MyOwnDataset(root='./datasets', city=opt.dataset, mode='train', generalization_test=True, norm_x=opt.norm_x)
    valid_data = MyOwnDataset(root='./datasets', city=opt.dataset, mode='valid', generalization_test=True, norm_x=opt.norm_x)
    test_data = MyOwnDataset(root='./datasets', city=opt.dataset, mode='test', generalization_test=True, norm_x=opt.norm_x)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size)
    valid_loader = DataLoader(valid_data, batch_size=opt.batch_size)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size)
    print("data loaded.")


    '''initiate model'''
    model = eval(opt.model_name)(dim_in=opt.dim_in)
    model.cuda(opt.gpu)

    '''initiate criteria and optimizer'''
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.l2)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

    # save dir
    log_dir = f"asset/log"
    save_dir = f"asset/model"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # save best metric
    losses = [np.inf]
    no_better_epoch = 0
    early_stop_epoch = 0
    best_metric = {'epoch':0, 'id-mse':0, 'id-rmse':0, 'id-mae':0, 'id-median':0, 'ood-mse':0, 'ood-rmse':0, 'ood-mae':0, 'ood-median':0}
    header = ['seed'] + list(best_metric.keys())
    log_file = os.path.join(log_dir, f'{opt.dataset}.csv')
    save_file = os.path.join(save_dir, f'{opt.dataset}_seed{opt.seed}_best.pth')
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)

    for epoch in range(1, opt.epochs+1):
        print("epoch {}.    ".format(epoch))
        total_mse, total_mae, train_num = 0, 0, 0
        model.train()
        a = []
        for i, batch in enumerate(train_loader):
            batch.cuda(opt.gpu)

            outputs = model(batch.x, batch.edge_index, batch.tg_mask, batch.batch, batch.ptr)
            y_range = batch.y_max[batch.tg_mask == 1] - batch.y_min[batch.tg_mask == 1]
            y = batch.y[batch.tg_mask == 1]

            # loss function
            perm_mse = get_mselist(y, outputs['perm_pred'], y_range)
            xc_mse = get_mselist(y, outputs['xc_pred'], y_range)
            distance = torch.sqrt(xc_mse)
            fea_reg = torch.sum(outputs['feature_mask']) * opt.fea
            edge_reg = bernoulli_loss(opt.fix_r, outputs['edge_att']) * opt.edge

            optimizer.zero_grad()
            loss = perm_mse.sum() + xc_mse.sum() + fea_reg + edge_reg
            loss.backward()
            optimizer.step()

            total_mse += xc_mse.sum()
            total_mae += distance.sum()
            train_num += len(y) 

        total_mse /= train_num
        total_mae /= train_num
        print("train: mse loss: {:.4f} mae: {:.4f}".format(total_mse, total_mae))


        with torch.no_grad():
        
            # valid
            v_mse, v_mae, v_num = 0, 0, 0
            v_distance = []
            for i, batch in enumerate(valid_loader):
                batch.cuda(opt.gpu)

                outputs = model(batch.x, batch.edge_index, batch.tg_mask, batch.batch, batch.ptr, training=False)
                y_range = batch.y_max[batch.tg_mask == 1] - batch.y_min[batch.tg_mask == 1]
                y = batch.y[batch.tg_mask == 1]

                mse = get_mselist(y, outputs['xc_pred'], y_range)
                distance = torch.sqrt(mse)
                for i in range(len(distance.cpu().detach().numpy())):
                    v_distance.append(distance.cpu().detach().numpy()[i])

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
            for i, batch in enumerate(test_loader):
                batch.cuda(opt.gpu)            

                outputs = model(batch.x, batch.edge_index, batch.tg_mask, batch.batch, batch.ptr, training=False)   
                y_range = batch.y_max[batch.tg_mask == 1] - batch.y_min[batch.tg_mask == 1]
                y = batch.y[batch.tg_mask == 1]        

                mse = get_mselist(y, outputs['xc_pred'], y_range)
                distance = torch.sqrt(mse)   
                for i in range(len(distance.cpu().detach().numpy())):
                    t_distance.append(distance.cpu().detach().numpy()[i])
                
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
                torch.save({'epoch': epoch, 'mae': batch_mae, 'model_state_dict': model.state_dict()}, save_file)
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