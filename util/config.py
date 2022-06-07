# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import numpy as np
import torch


def img_param_init(args):
    dataset = args.dataset
    if dataset == 'office':
        domains = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-caltech':
        domains = ['amazon', 'dslr', 'webcam', 'caltech']
    elif dataset == 'off_home':
        domains = ['Art', 'Clipart', 'Product', 'Real_World']
    elif dataset == 'dg5':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
    elif dataset == 'pacs':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset == 'vlcs':
        domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    elif dataset == 'dg4':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn']
    elif dataset == 'terra':
        domains = ['location_38', 'location_43', 'location_46', 'location_100']
    elif dataset == 'domain_net':
        domains = ["clipart", "infograph",
                   "painting", "quickdraw", "real", "sketch"]
    else:
        print('No such dataset exists!')
    args.domains = domains
    args.img_dataset = {
        'office': ['amazon', 'dslr', 'webcam'],
        'office-caltech': ['amazon', 'dslr', 'webcam', 'caltech'],
        'off_home': ['Art', 'Clipart', 'Product', 'Real_World'],
        'pacs': ['art_painting', 'cartoon', 'photo', 'sketch'],
        'dg5': ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'],
        'vlcs': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],
        'dg4': ['mnist', 'mnist_m', 'svhn', 'syn'],
        'terra': ['location_38', 'location_43', 'location_46', 'location_100'],
        'domain_net': ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    }
    if dataset in ['dg5', 'dg4']:
        args.shuffle_shape = (3, 36, 36)
        args.input_shape = (3, 32, 32)
        args.num_classes = 10
    else:
        args.shuffle_shape = (3, 228, 228)
        args.input_shape = (3, 224, 224)
        if args.dataset == 'off_home':
            args.num_classes = 65
        elif args.dataset == 'office':
            args.num_classes = 31
        elif args.dataset == 'pacs':
            args.num_classes = 7
        elif args.dataset == 'vlcs':
            args.num_classes = 5
        elif args.dataset == 'terra':
            args.num_classes = 10
        elif args.dataset == 'domain_net':
            args.num_classes = 345
        else:
            args.num_classes = 4
    return args


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
