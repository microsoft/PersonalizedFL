# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import torch
import argparse
import numpy as np

from datautil.prepare_data import *
from util.config import img_param_init, set_random_seed
from alg import algs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='Algorithm to choose: [base | fedavg | fedbn | fedprox | fedap]')
    parser.add_argument('--datapercent', type=float,
                        default=1e-1, help='data percent to use')
    parser.add_argument('--dataset', type=str, default='pacs',
                        help='[vlcs | pacs | officehome | pamap | covid | medmnist]')
    parser.add_argument('--root_dir', type=str,
                        default='./data/', help='data path')
    parser.add_argument('--save_path', type=str,
                        default='./cks/', help='path to save the checkpoint')
    parser.add_argument('--device', type=str,
                        default='cuda', help='[cuda | cpu]')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--iters', type=int, default=300,
                        help='iterations for communication')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--n_clients', type=int,
                        default=20, help='number of clients')
    parser.add_argument('--non_iid_alpha', type=float,
                        default=0.1, help='data split for label shift')
    parser.add_argument('--partition_data', type=str,
                        default='non_iid_dirichlet', help='partition data way')
    parser.add_argument('--pretrained_iters', type=int,
                        default=150, help='iterations for pretrained models')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')

    # algorithm-specific parameters
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='The hyper parameter for fedprox')
    parser.add_argument('--model_momentum', type=float,
                        default=0.5, help='hyperparameter for fedap')
    args = parser.parse_args()

    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)

    if args.dataset in ['vlcs', 'pacs', 'off_home']:
        args = img_param_init(args)
        args.n_clients = 4

    exp_folder = f'fed_{args.dataset}_{args.alg}_{args.datapercent}_{args.non_iid_alpha}_{args.mu}_{args.model_momentum}_{args.iters}_{args.wk_iters}'
    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, args.alg)

    train_loaders, val_loaders, test_loaders = get_data(args.dataset)(args)

    algclass = algs.get_algorithm_class(args.alg)(args)

    if args.alg == 'fedap':
        algclass.set_client_weight(train_loaders)

    best_changed = False

    best_acc = [0] * args.n_clients
    best_tacc = [0] * args.n_clients
    start_iter = 0

    for a_iter in range(start_iter, args.iters):
        print(f"============ Train round {a_iter} ============")

        # local client training
        for wi in range(args.wk_iters):
            for client_idx in range(args.n_clients):
                algclass.client_train(
                    client_idx, train_loaders[client_idx], a_iter)

        # server aggregation
        algclass.server_aggre()
        
        # evaluation on training data
        for client_idx in range(args.n_clients):
            train_loss, train_acc = algclass.client_eval(
                client_idx, train_loaders[client_idx])
            print(f' Site-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')

        # evaluation on valid data
        val_acc_list = [None] * args.n_clients
        for client_idx in range(args.n_clients):
            val_loss, val_acc = algclass.client_eval(
                client_idx, val_loaders[client_idx])
            val_acc_list[client_idx] = val_acc
            print(f' Site-{client_idx:02d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        if np.mean(val_acc_list) > np.mean(best_acc):
            for client_idx in range(args.n_clients):
                best_acc[client_idx] = val_acc_list[client_idx]
                best_epoch = a_iter
            best_changed = True

        if best_changed:
            print(f' Saving the local and server checkpoint to {SAVE_PATH}')
            tosave = {'model': algclass.state_dict(
            ), 'best_epoch': best_epoch, 'best_acc': best_acc}
            torch.save(tosave, SAVE_PATH)
            best_changed = False
            # test
            for client_idx in range(args.n_clients):
                _, test_acc = algclass.client_eval(
                    client_idx, test_loaders[client_idx])
                print(
                    f' Test site-{client_idx:02d} | Epoch:{best_epoch} | Test Acc: {test_acc:.4f}')
                best_tacc[client_idx] = test_acc

    s = 'Personalized test acc for each client: '
    for item in best_tacc:
        s += f'{item:.4f},'
    mean_acc_test = np.mean(np.array(best_tacc))
    s += f'\nAverage accuracy: {mean_acc_test:.4f}'
    print(s)
