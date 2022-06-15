#coding=utf-8
from cgi import test
import os,sys
sys.path.append('/home/lw/lw/PersonalizedFL')
import numpy as np
import torch
import argparse
import numpy as np


from datautil.prepare_data import *
from util.config import img_param_init, set_random_seed
from util.evalandprint import evalandprint
from alg import algs

def initmodel():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='Algorithm to choose: [base | fedavg | fedbn | fedprox | fedap | metafed ]')
    parser.add_argument('--datapercent', type=float,
                        default=1e-1, help='data percent to use')
    parser.add_argument('--dataset', type=str, default='medmnist',
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
    parser.add_argument('--plan', type=int,
                        default=1, help='choose the feature type')
    parser.add_argument('--pretrained_iters', type=int,
                        default=150, help='iterations for pretrained models')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--nosharebn', action='store_true',
                        help='not share bn')

    # algorithm-specific parameters
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='The hyper parameter for fedprox')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='threshold to use copy or distillation, hyperparmeter for metafed')
    parser.add_argument('--lam', type=float, default=1.0,
                        help='init lam, hyperparmeter for metafed')
    parser.add_argument('--model_momentum', type=float,
                        default=0.5, help='hyperparameter for fedap')
    args = parser.parse_args()

    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)
    
    if args.dataset in ['vlcs', 'pacs', 'off_home']:
        args = img_param_init(args)
        args.n_clients = 4
    return args


def getres(alg,dataset,noniid=0.1):
    rwl=[[300,1],[100,3],[50,6],[30,10],[5,60],[3,100]]
    s=[]
    s1=[]

    if alg in ['fedprox']:
        paraml=[0.01,0.1,1,10]
        for rw in rwl:
            for item in paraml:
                s.append('fed_'+dataset+'_'+alg+'_0.1_'+str(noniid)+'_'+str(float(item))+'_0.5_'+str(rw[0])+'_'+str(rw[1]))
                s1.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(rw[1])+' --non_iid_alpha '+str(noniid)+' --mu '+str(item))
    elif alg in ['fedap']:
        paraml=[0.1,0.3,0.5,0.7,0.9]
        for rw in rwl:
            for item in paraml:
                s.append('fed_'+dataset+'_'+alg+'_0.1_'+str(noniid)+'_0.001_'+str(item)+'_'+str(rw[0])+'_'+str(rw[1]))
                s1.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(rw[1])+' --non_iid_alpha '+str(noniid)+' --model_momentum '+str(item))
    elif alg in ['metafed']:
        for rw in rwl:
            for threshold in [0.0, 0.4, 0.5, 0.6, 1.1]:
                if threshold > 1:
                    s1.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(
                        rw[1])+' --threshold '+str(threshold)+' --non_iid_alpha '+str(noniid))
                    s1.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(
                        rw[1])+' --threshold '+str(threshold)+' --nosharebn'+' --non_iid_alpha '+str(noniid))
                    s.append('fed_'+dataset+'_'+alg+'_0.1_'+str(noniid)+'_0.001_0.5_'+str(1)+'_'+str(1.0)+'_'+str(threshold)+'_'+str(rw[0])+'_'+str(rw[1]))    
                    s.append('fed_'+dataset+'_'+alg+'_0.1_'+str(noniid)+'_0.001_0.5_'+str(1)+'_'+str(1.0)+'_'+str(threshold)+'_'+str(rw[0])+'_'+str(rw[1])+'_nosharebn')
                else:
                    for lam in [0.1, 1.0, 5.0, 10.0]:
                        for plan in [0, 1, 2]:
                            s1.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(rw[1])+' --lam '+str(
                                lam)+' --threshold '+str(threshold)+' --plan '+str(plan)+' --non_iid_alpha '+str(noniid))
                            s1.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(rw[1])+' --lam '+str(
                                lam)+' --threshold '+str(threshold)+' --plan '+str(plan)+' --nosharebn'+' --non_iid_alpha '+str(noniid))     
                            s.append('fed_'+dataset+'_'+alg+'_0.1_'+str(noniid)+'_0.001_0.5_'+str(plan)+'_'+str(lam)+'_'+str(threshold)+'_'+str(rw[0])+'_'+str(rw[1]))       
                            s.append('fed_'+dataset+'_'+alg+'_0.1_'+str(noniid)+'_0.001_0.5_'+str(plan)+'_'+str(lam)+'_'+str(threshold)+'_'+str(rw[0])+'_'+str(rw[1])+'_nosharebn')
    else:
        for rw in rwl:
            s.append('fed_'+dataset+'_'+alg+'_0.1_'+str(noniid)+'_0.001_0.5_'+str(rw[0])+'_'+str(rw[1]))    
            s1.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(rw[1])+' --non_iid_alpha '+str(noniid))    


    bes=0
    tf,tts=0,0
    notfin1=''
    notfin2=''
    for i,item in enumerate(s):
        filename='./cks/'+item+'/done.txt'
        try:
            with open(filename,'r') as f:
                ts=float(f.read().split(',')[-1])
                if ts>bes:
                    bes=ts
                    tf=item
                    tts=s1[i]
        except Exception as e:
            try:
                args=initmodel()
                args.alg='metafed'
                # args.dataset='medmnist'
                train_loaders, val_loaders, test_loaders = get_data(args.dataset)(args)
                algclass = algs.get_algorithm_class(args.alg)(args).cuda()
                # tacc=0
                # tmp=[]
                # for client_idx in range(args.n_clients):
                #     _, test_acc = algclass.client_eval(client_idx, test_loaders[client_idx])
                #     tacc+=test_acc
                #     tmp.append(test_acc)
                # print(tmp)
                te=torch.load('./cks/'+item+'/metafed')
                algclass.load_state_dict(te['model'])
                # print(te['best_epoch'],te['best_acc'])
                print(te['model'].keys())
                print(algclass)
                print(algclass.state_dict())
                tacc=0
                tmp=[]
                with torch.no_grad():
                    for client_idx in range(args.n_clients):
                        _, test_acc = algclass.client_eval(client_idx, test_loaders[client_idx])
                        tacc+=test_acc
                        tmp.append(test_acc)
                print(tmp)
                ts=tacc/args.n_clients
                if ts>bes:
                    bes=ts
                    tf=item
                    tts=s1[i]
            except Exception as e:
                notfin1+=(item+'\n')
                notfin2+=(s1[i]+'\n')
    if len(notfin1)>0:
        with open('./script/notfinfile.txt','w') as f:
            f.write(notfin1)
        with open('./script/notfinindex.txt','w') as f:
            f.write(notfin2)        
    return bes,tf,tts

s=[]
for data in ['medmnist']:
    for noni in [0.1,0.01]:
        # for alg in ['base','fedavg','fedprox','fedbn','fedap']:
        for alg in ['metafed']:
            print(getres(alg,data,noni))
        #    print(alg,getres(alg,data,noni))
        #    print(getres(alg,data,noni)[2])
