from torch.utils.data import Dataset
from typing import Tuple, Any
import random


class DistributedDataset(Dataset):
    def __init__(self, dataset: Dataset, index):
        super().__init__()
        self.dataset = dataset
        self.index = index

    def __getitem__(self, item):
        return self.dataset.__getitem__(self.index[item])

    def __len__(self):
        return len(self.index)


def distributed_dataset(dataset: Dataset, split: Any, rank: int, size: int = None, seed: int = 777):
    if size is None:
        size = len(dataset)
    random.seed(seed)
    indexes = [x for x in range(size)]
    random.shuffle(indexes)
    indexes_list = []
    for s in split:
        indexes_list.append(indexes[:int(s * size)])
        indexes = indexes[int(s * size):]
    return DistributedDataset(dataset, indexes_list[rank])
