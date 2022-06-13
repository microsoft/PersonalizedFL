# coding=utf-8
import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import subprocess
import time
import argparse

def multi_gpu_launcher(commands, task_num_per_gpu=4, gpulist=[]):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """

    n_gpus = len(gpulist)
    gpulist = gpulist*task_num_per_gpu
    procs_by_gpu = [None]*(n_gpus*task_num_per_gpu)
    # print(gpulist)
    while len(commands) > 0:
        for idx in range(n_gpus*task_num_per_gpu):
            proc = procs_by_gpu[idx]
            # 前一个None表示还没有开始
            # 后一个表示如果None为正在运行，否则是运行完成
            if ((proc is None) or (proc.poll() is not None)) and (len(commands) > 0):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpulist[idx]} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

def readfromfile(filename,st,en):
    with open(filename,'r') as f:
        s=f.read()
    ss=s.split('\n')
    return ss[:-1][st:en]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type = str, default='ours.txt', help='filename')
    parser.add_argument('--st', type = int, default=0, help='start')
    parser.add_argument('--en', type = int, default=0, help='end')
    parser.add_argument('--pg', type = int, default=1, help='per gpu')
    args = parser.parse_args()

    cmdlist=readfromfile(args.filename,args.st,args.en)
    # cmdlist=readfromfile('../script/tmp/sc/off-cal.txt',28,32)
    print(len(cmdlist))
    print(cmdlist[0])
    try:
        multi_gpu_launcher(cmdlist,args.pg, [0,1])
    except Exception as e:
        pass