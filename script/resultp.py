#coding=utf-8

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
            for threshold in [0, 0.4, 0.5, 0.6, 1.1]:
                if threshold > 1:
                    s1.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(
                        rw[1])+' --threshold '+str(threshold)+' --non_iid_alpha '+str(noniid))
                    s1.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(
                        rw[1])+' --threshold '+str(threshold)+' --nosharebn'+' --non_iid_alpha '+str(noniid))
                    s.append('fed_'+dataset+'_'+alg+'_0.1_'+str(noniid)+'_0.001_0.5_'+str(1)+'_'+str(1.0)+'_'+str(threshold)+'_'+str(rw[0])+'_'+str(rw[1]))    
                    s.append('fed_'+dataset+'_'+alg+'_0.1_'+str(noniid)+'_0.001_0.5_'+str(1)+'_'+str(1.0)+'_'+str(threshold)+'_'+str(rw[0])+'_'+str(rw[1]))+'_nosharebn'
                else:
                    for lam in [0.1, 1, 5, 10]:
                        for plan in [0, 1, 2]:
                            s1.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(rw[1])+' --lam '+str(
                                lam)+' --threshold '+str(threshold)+' --plan '+str(plan)+' --non_iid_alpha '+str(noniid))
                            s1.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(rw[1])+' --lam '+str(
                                lam)+' --threshold '+str(threshold)+' --plan '+str(plan)+' --nosharebn'+' --non_iid_alpha '+str(noniid))     
                            s.append('fed_'+dataset+'_'+alg+'_0.1_'+str(noniid)+'_0.001_0.5_'+str(plan)+'_'+str(lam)+'_'+str(threshold)+'_'+str(rw[0])+'_'+str(rw[1]))       
                            s.append('fed_'+dataset+'_'+alg+'_0.1_'+str(noniid)+'_0.001_0.5_'+str(plan)+'_'+str(lam)+'_'+str(threshold)+'_'+str(rw[0])+'_'+str(rw[1]))+'_nosharebn'
    else:
        for rw in rwl:
            s.append('fed_'+dataset+'_'+alg+'_0.1_'+str(noniid)+'_0.001_0.5_'+str(rw[0])+'_'+str(rw[1]))    
            s1.append('python main.py --alg '+alg+' --dataset '+dataset+' --iters '+str(rw[0])+' --wk_iters '+str(rw[1])+' --non_iid_alpha '+str(noniid))    


    bes=0
    tf,tts=0,0
    for i,item in enumerate(s):
        filename='./cks/'+item+'/done.txt'
        with open(filename,'r') as f:
            ts=float(f.read().split(',')[-1])
            if ts>bes:
                bes=ts
                tf=item
                tts=s1[i]
    return bes,tf,tts

s=[]
for data in ['medmnist']:
    for noni in [0.1,0.01]:
        for alg in ['base','fedavg','fedprox','fedbn','fedap']:
        #    print(alg,getres(alg,data,noni))
           print(getres(alg,data,noni)[2])
