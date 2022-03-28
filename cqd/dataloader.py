# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset
from util import flatten


class CQDTrainDataset(Dataset):
    def __init__(self, queries, nentity, nrelation, negative_sample_size, answer):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.answer = answer

        self.qa_lst = []
        for q, qs in queries:
            for a in self.answer[q]:
                qa_entry = (qs, q, a)
                self.qa_lst += [qa_entry]

        self.qa_len = len(self.qa_lst)

    def __len__(self):
        # return self.len
        return self.qa_len

    def __getitem__(self, idx):
        # query = self.queries[idx][0]
        query = self.qa_lst[idx][1]

        # query_structure = self.queries[idx][1]
        query_structure = self.qa_lst[idx][0]

        # tail = np.random.choice(list(self.answer[query]))
        tail = self.qa_lst[idx][2]

        # subsampling_weight = self.count[query]
        # subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        subsampling_weight = torch.tensor([1.0])

        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.answer[query],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor([tail])
        return positive_sample, negative_sample, subsampling_weight, flatten(query), query_structure

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        query = [_[3] for _ in data]
        query_structure = [_[4] for _ in data]
        return positive_sample, negative_sample, subsample_weight, query, query_structure
