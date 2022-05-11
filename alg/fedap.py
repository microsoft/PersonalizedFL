import os
import copy
import math
import numpy as np
import torch

from alg.fedavg import fedavg
from util.traineval import pretrain_model


class fedap(fedavg):
    def __init__(self, args):
        super(fedap, self).__init__(args)

    def set_client_weight(self, train_loaders):
        os.makedirs('./checkpoint/'+'pretrained/', exist_ok=True)
        preckpt = './checkpoint/'+'pretrained/' + \
            self.args.dataset+'_'+str(self.args.batch)
        self.pretrain_model = copy.deepcopy(
            self.server_model).to(self.args.device)
        if not os.path.exists(preckpt):
            pretrain_model(self.args, self.pretrain_model,
                           preckpt, self.args.device)
        self.preckpt = preckpt
        self.client_weight = get_weight_preckpt(
            self.args, self.pretrain_model, self.preckpt, train_loaders, self.client_weight)


def get_form(model):
    tmpm = []
    tmpv = []
    for name in model.state_dict().keys():
        if 'running_mean' in name:
            tmpm.append(model.state_dict()[name].detach().to('cpu').numpy())
        if 'running_var' in name:
            tmpv.append(model.state_dict()[name].detach().to('cpu').numpy())
    return tmpm, tmpv


def get_wasserstein(m1, v1, m2, v2, mode='nosquare'):
    w = 0
    bl = len(m1)
    for i in range(bl):
        tw = 0
        tw += (np.sum(np.square(m1[i]-m2[i])))
        tw += (np.sum(np.square(np.sqrt(v1[i]) - np.sqrt(v2[i]))))
        if mode == 'square':
            w += tw
        else:
            w += math.sqrt(tw)
    return w


def get_weight_matrix1(args, bnmlist, bnvlist, client_weights):
    client_num = len(bnmlist)
    weight_m = np.zeros((client_num, client_num))
    for i in range(client_num):
        for j in range(client_num):
            if i == j:
                weight_m[i, j] = 0
            else:
                tmp = get_wasserstein(
                    bnmlist[i], bnvlist[i], bnmlist[j], bnvlist[j])
                if tmp == 0:
                    weight_m[i, j] = 100000000000000
                else:
                    weight_m[i, j] = 1/tmp
    weight_s = np.sum(weight_m, axis=1)
    weight_s = np.repeat(weight_s, client_num).reshape(
        (client_num, client_num))
    weight_m = (weight_m/weight_s)*(1-args.model_momentum)
    for i in range(client_num):
        weight_m[i, i] = args.model_momentum
    return weight_m


def get_weight_preckpt(args, model, preckpt, trainloadrs, client_weights, device='cuda'):
    model.load_state_dict(torch.load(preckpt)['state'])
    model.eval()
    bnmlist1, bnvlist1 = [], []
    for i in range(args.n_clients):
        avgmeta = metacount(get_form(model)[0])
        with torch.no_grad():
            for data, _ in trainloadrs[i]:
                data = data.to(device).float()
                fea = model.getallfea(data)
                nl = len(data)
                tm, tv = [], []
                for item in fea:
                    if len(item.shape) == 4:
                        tm.append(torch.mean(
                            item, dim=[0, 2, 3]).detach().to('cpu').numpy())
                        tv.append(
                            torch.var(item, dim=[0, 2, 3]).detach().to('cpu').numpy())
                    else:
                        tm.append(torch.mean(
                            item, dim=0).detach().to('cpu').numpy())
                        tv.append(
                            torch.var(item, dim=0).detach().to('cpu').numpy())
                avgmeta.update(nl, tm, tv)
        bnmlist1.append(avgmeta.getmean())
        bnvlist1.append(avgmeta.getvar())
    weight_m = get_weight_matrix1(args, bnmlist1, bnvlist1, client_weights)
    return weight_m


class metacount(object):
    def __init__(self, numpyform):
        super(metacount, self).__init__()
        self.count = 0
        self.mean = []
        self.var = []
        self.bl = len(numpyform)
        for i in range(self.bl):
            self.mean.append(np.zeros(len(numpyform[i])))
            self.var.append(np.zeros(len(numpyform[i])))

    def update(self, m, tm, tv):
        tmpcount = self.count+m
        for i in range(self.bl):
            tmpm = (self.mean[i]*self.count + tm[i]*m)/tmpcount
            self.var[i] = (self.count*(self.var[i]+np.square(tmpm -
                           self.mean[i])) + m*(tv[i]+np.square(tmpm-tm[i])))/tmpcount
            self.mean[i] = tmpm
        self.count = tmpcount

    def getmean(self):
        return self.mean

    def getvar(self):
        return self.var
