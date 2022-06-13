# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8

def gensc(alg, dataset, noniid=0.1):
    # pass
    rwl = [[300, 1], [100, 3], [50, 6], [30, 10], [5, 60], [3, 100]]

    s = []

    if alg in ['fedprox']:
        paraml = [0.01, 0.1, 1, 10]
        for rw in rwl:
            for item in paraml:
                s.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(
                    rw[0])+' --wk_iters '+str(rw[1])+' --non_iid_alpha '+str(noniid)+' --mu '+str(item))
    elif alg in ['fedap']:
        paraml = [0.1, 0.3, 0.5, 0.7, 0.9]
        for rw in rwl:
            for item in paraml:
                s.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(
                    rw[0])+' --wk_iters '+str(rw[1])+' --non_iid_alpha '+str(noniid)+' --model_momentum '+str(item))
    elif alg in ['metafed']:
        for rw in rwl:
            for threshold in [0, 0.4, 0.5, 0.6, 1.1]:
                if threshold > 1:
                    s.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(
                        rw[1])+' --threshold '+str(threshold)+' --non_iid_alpha '+str(noniid))
                    s.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(
                        rw[1])+' --threshold '+str(threshold)+' --nosharebn'+' --non_iid_alpha '+str(noniid))
                else:
                    for lam in [0.1, 1, 5, 10]:
                        for plan in [0, 1, 2]:
                            s.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(rw[1])+' --lam '+str(
                                lam)+' --threshold '+str(threshold)+' --plan '+str(plan)+' --non_iid_alpha '+str(noniid))
                            s.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(rw[1])+' --lam '+str(
                                lam)+' --threshold '+str(threshold)+' --plan '+str(plan)+' --nosharebn'+' --non_iid_alpha '+str(noniid))
    else:
        for rw in rwl:
            s.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters ' +
                     str(rw[0])+' --wk_iters '+str(rw[1])+' --non_iid_alpha '+str(noniid))

    return s


s = []
# for data in ['medmnist', 'pacs']:
for data in ['medmnist']:
    for noni in [0.1, 0.01]:
        # for alg in ['base', 'fedavg', 'fedprox', 'fedbn', 'fedap']:
        for alg in ['metafed']:
            s += gensc(alg, data, noni)

ts = ''
for item in s:
    ts += item+'\n'

with open('script/tmp/testmeta.txt', 'w') as f:
    f.write(ts)
