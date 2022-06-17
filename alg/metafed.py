# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.optim as optim
import copy

from util.modelsel import modelsel
from util.traineval import trainwithteacher, test


class metafed(torch.nn.Module):
    def __init__(self, args):
        super(metafed, self).__init__()
        self.server_model, self.client_model, self.client_weight = modelsel(
            args, args.device)
        self.optimizers = [optim.SGD(params=self.client_model[idx].parameters(
        ), lr=args.lr) for idx in range(args.n_clients)]
        self.loss_fun = nn.CrossEntropyLoss()
        args.sort = ''
        for i in range(args.n_clients):
            args.sort += '%d-' % i
        args.sort = args.sort[:-1]
        self.args = args
        self.csort = [int(item) for item in args.sort.split('-')]

    def init_model_flag(self, train_loaders, val_loaders):
        self.flagl = []
        client_num = self.args.n_clients
        for _ in range(client_num):
            self.flagl.append(False)
        optimizers = [optim.SGD(params=self.client_model[idx].parameters(
        ), lr=self.args.lr) for idx in range(client_num)]
        for idx in range(client_num):
            client_idx = idx
            model, train_loader, optimizer, tmodel, val_loader = self.client_model[
                client_idx], train_loaders[client_idx], optimizers[client_idx], None, val_loaders[idx]
            for _ in range(30):
                _, _ = trainwithteacher(
                    model, train_loader, optimizer, self.loss_fun, self.args.device, tmodel, 1, self.args, False)
            _, val_acc = test(model, val_loader,
                              self.loss_fun, self.args.device)
            if val_acc > self.args.threshold:
                self.flagl[idx] = True
        if self.args.dataset in ['vlcs', 'pacs']:
            self.thes = 0.4
        elif 'medmnist' in self.args.dataset:
            self.thes = 0.5
        elif 'pamap' in self.args.dataset:
            self.thes = 0.5
        else:
            self.thes = 0.5

    def update_flag(self, val_loaders):
        for client_idx, model in enumerate(self.client_model):
            _, val_acc = test(
                model, val_loaders[client_idx], self.loss_fun, self.args.device)
            if val_acc > self.args.threshold:
                self.flagl[client_idx] = True

    def client_train(self, c_idx, dataloader, round):
        client_idx = self.csort[c_idx]
        tidx = self.csort[c_idx-1]
        model, train_loader, optimizer, tmodel = self.client_model[
            client_idx], dataloader, self.optimizers[client_idx], self.client_model[tidx]
        if round == 0 and c_idx == 0:
            tmodel = None
        for _ in range(self.args.wk_iters):
            train_loss, train_acc = trainwithteacher(
                model, train_loader, optimizer, self.loss_fun, self.args.device, tmodel, self.args.lam, self.args, self.flagl[client_idx])
        return train_loss, train_acc

    def personalization(self, c_idx, dataloader, val_loader):
        client_idx = self.csort[c_idx]
        model, train_loader, optimizer, tmodel = self.client_model[
            client_idx], dataloader, self.optimizers[client_idx], copy.deepcopy(self.client_model[self.csort[-1]])

        with torch.no_grad():
            _, v1a = test(model, val_loader, self.loss_fun, self.args.device)
            _, v2a = test(tmodel, val_loader, self.loss_fun, self.args.device)

        if v2a <= v1a and v2a < self.thes:
            lam = 0
        else:
            lam = (10**(min(1, (v2a-v1a)*5)))/10*self.args.lam

        for _ in range(self.args.wk_iters):
            train_loss, train_acc = trainwithteacher(
                model, train_loader, optimizer, self.loss_fun, self.args.device, tmodel, lam, self.args, self.flagl[client_idx])
        return train_loss, train_acc

    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc = test(
            self.client_model[c_idx], dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc
