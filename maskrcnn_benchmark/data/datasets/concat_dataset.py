# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the sizes of the image
    """

    def get_idxs(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_img_info(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_img_info(sample_idx)

class MixDataset(object):
    def __init__(self, datasets, ratios):
        self.datasets = datasets
        self.ratios = ratios
        self.lengths = []
        for dataset in self.datasets:
            self.lengths.append(len(dataset))
        self.lengths = np.array(self.lengths)
        self.seperate_inds = []
        s = 0
        for i in self.ratios[:-1]:
            s += i
            self.seperate_inds.append(s)

    def __len__(self):
       return self.lengths.sum()
       
    def __getitem__(self, item):
        i = np.random.rand()
        ind = bisect.bisect_right(self.seperate_inds, i)
        b_ind = np.random.randint(self.lengths[ind])
        return self.datasets[ind][b_ind]








