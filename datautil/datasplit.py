import numpy as np
import torch
import os
import math
import functools
import torch.distributed as dist


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices
        self.replaced_targets = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        return (self.data[data_idx][0], self.data[data_idx][1])

    def update_replaced_targets(self, replaced_targets):
        self.replaced_targets = replaced_targets

        # evaluate the the difference between original labels and the simulated labels.
        count = 0
        for index in range(len(replaced_targets)):
            data_idx = self.indices[index]

            if self.replaced_targets[index] == self.data[data_idx][1]:
                count += 1
        return count / len(replaced_targets)

    def set_targets(self, replaced_targets):
        self.replaced_targets = replaced_targets

    def get_targets(self):
        return self.replaced_targets

    def clean_replaced_targets(self):
        self.replaced_targets = None


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(
        self, conf, data, partition_sizes, partition_type, consistent_indices=True
    ):
        # prepare info.
        self.conf = conf
        self.partition_sizes = partition_sizes
        self.partition_type = partition_type
        self.consistent_indices = consistent_indices
        self.partitions = []

        # get data, data_size, indices of the data.
        self.data_size = len(data)
        if type(data) is not Partition:
            self.data = data
            indices = np.array([x for x in range(0, self.data_size)])
        else:
            self.data = data.data
            indices = data.indices

        self.partition_indices(indices)

    def partition_indices(self, indices):
        indices = self._create_indices(indices)
        if self.consistent_indices:
            indices = self._get_consistent_indices(indices)

        if self.partition_type == 'evenly':
            classes = np.unique(self.data.targets)
            lp = len(self.partition_sizes)
            ti = indices[:, 0]
            ttar = indices[:, 1]
            for i in range(lp):
                self.partitions.append(np.array([]))
            for c in classes:
                tindice = np.where(ttar == c)[0]
                lti = len(tindice)
                from_index = 0
                for i in range(lp):
                    partition_size = self.partition_sizes[i]
                    to_index = from_index + int(partition_size * lti)
                    if i == (lp-1):
                        self.partitions[i] = np.hstack(
                            (self.partitions[i], ti[tindice[from_index:]]))
                    else:
                        self.partitions[i] = np.hstack(
                            (self.partitions[i], ti[tindice[from_index:to_index]]))
                    from_index = to_index
            for i in range(lp):
                self.partitions[i] = self.partitions[i].astype(np.int).tolist()
        else:
            from_index = 0
            for partition_size in self.partition_sizes:
                to_index = from_index + int(partition_size * self.data_size)
                self.partitions.append(indices[from_index:to_index])
                from_index = to_index

        record_class_distribution(
            self.partitions, self.data.targets
        )

    def _create_indices(self, indices):
        if self.partition_type == "origin":
            pass
        elif self.partition_type == "random":
            # it will randomly shuffle the indices.
            self.conf.random_state.shuffle(indices)
        elif self.partition_type == 'evenly':
            indices = np.array([
                (idx, target)
                for idx, target in enumerate(self.data.targets)
                if idx in indices
            ])
        elif self.partition_type == "sorted":
            # it will sort the indices based on the data label.
            indices = [
                i[0]
                for i in sorted(
                    [
                        (idx, target)
                        for idx, target in enumerate(self.data.targets)
                        if idx in indices
                    ],
                    key=lambda x: x[1],
                )
            ]
        elif self.partition_type == "non_iid_dirichlet":
            num_classes = len(np.unique(self.data.targets))
            num_indices = len(indices)
            n_workers = len(self.partition_sizes)

            list_of_indices = build_non_iid_by_dirichlet(
                random_state=self.conf.random_state,
                indices2targets=np.array(
                    [
                        (idx, target)
                        for idx, target in enumerate(self.data.targets)
                        if idx in indices
                    ]
                ),
                non_iid_alpha=self.conf.non_iid_alpha,
                num_classes=num_classes,
                num_indices=num_indices,
                n_workers=n_workers,
            )
            indices = functools.reduce(lambda a, b: a + b, list_of_indices)
        else:
            raise NotImplementedError(
                f"The partition scheme={self.partition_type} is not implemented yet"
            )
        return indices

    def _get_consistent_indices(self, indices):
        if dist.is_initialized():
            # sync the indices over clients.
            indices = torch.IntTensor(indices)
            dist.broadcast(indices, src=0)
            return list(indices)
        else:
            return indices

    def use(self, partition_ind):
        return Partition(self.data, self.partitions[partition_ind])


def build_non_iid_by_dirichlet(
    random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    """
    refer to https://github.com/epfml/quasi-global-momentum/blob/3603211501e376d4a25fb2d427c30647065de8c8/code/pcode/datasets/partition_data.py
    """
    n_auxi_workers = 2
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, _ in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index: (num_indices if idx ==
                             num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch


def record_class_distribution(partitions, targets):
    targets_of_partitions = {}
    targets_np = np.array(targets)
    for idx, partition in enumerate(partitions):
        unique_elements, counts_elements = np.unique(
            targets_np[partition], return_counts=True
        )
        targets_of_partitions[idx] = list(
            zip(unique_elements, counts_elements))
    return targets_of_partitions


def define_val_dataset(conf, train_dataset):
    partition_sizes = [
        0.4, 0.3, 0.3
    ]
    data_partitioner = DataPartitioner(
        conf,
        train_dataset,
        partition_sizes,
        partition_type="evenly",
        # consistent_indices=False,
    )
    return data_partitioner


def define_data_loader(conf, dataset, data_partitioner=None):
    world_size = conf.n_clients
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    if data_partitioner is None:
        # update the data_partitioner.
        data_partitioner = DataPartitioner(
            conf, dataset, partition_sizes, partition_type=conf.partition_data
        )
    return data_partitioner


def getdataloader(conf, dataall, root_dir='./split/'):
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(root_dir+conf.dataset+str(conf.datapercent), exist_ok=True)
    file = root_dir+conf.dataset+str(conf.datapercent)+'/partion_'+conf.partition_data + \
        '_'+str(conf.non_iid_alpha)+'_'+str(conf.n_clients)+'.npy'
    if not os.path.exists(file):
        data_part = define_data_loader(conf, dataall)
        tmparr = []
        for i in range(conf.n_clients):
            tmppart = define_val_dataset(conf, data_part.use(i))
            tmparr.append(tmppart.partitions[0])
            tmparr.append(tmppart.partitions[1])
            tmparr.append(tmppart.partitions[2])
        tmparr = np.array(tmparr)
        np.save(file, tmparr)
    else:
        conf.partition_data = 'origin'
        data_part = define_data_loader(conf, dataall)
    data_part.partitions = np.load(file, allow_pickle=True).tolist()
    clienttrain_list = []
    clientvalid_list = []
    clienttest_list = []
    for i in range(conf.n_clients):
        clienttrain_list.append(data_part.use(3*i))
        clienttest_list.append(data_part.use(3*i+1))
        clientvalid_list.append(data_part.use(3*i+2))
    return clienttrain_list, clientvalid_list, clienttest_list


def define_pretrain_dataset(conf, train_dataset):
    partition_sizes = [
        0.3, 0.7
    ]
    data_partitioner = DataPartitioner(
        conf,
        train_dataset,
        partition_sizes,
        partition_type="evenly",
        # consistent_indices=False,
    )
    return data_partitioner.use(0)
