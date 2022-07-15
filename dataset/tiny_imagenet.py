from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import pickle
from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from .distributed_dataset import distributed_dataset


class TinyImageNet(Dataset):
    def __init__(self, root='./data', train=True, transform=None):
        root = os.path.join(root, 'tiny-imagenet')
        if train:
            root = os.path.join(root, 'tiny-imagenet_train.pkl')
        else:
            root = os.path.join(root, 'tiny-imagenet_val.pkl')
        with open(root, 'rb') as f:
            dat = pickle.load(f)
        self.data = dat['data']
        self.targets = dat['targets']
        self.transform = transform

    def __getitem__(self, item):
        data, targets = Image.fromarray(self.data[item]), self.targets[item]
        if self.transform is not None:
            data = self.transform(data)
        return data, targets

    def __len__(self):
        return len(self.data)


def tiny_imagenet(rank, split=None,
                  batch_size=None, transforms=None
                  , test_batch_size=64, is_distribute=True, seed=777, **kwargs):
    if transforms is None:
        transforms = tfs.Compose([
            tfs.ToTensor(),
        ])
    if batch_size is None:
        batch_size = 1
    if split is None:
        split = [1.0]
    train_set = TinyImageNet("../data", True, transforms)
    test_set = TinyImageNet("../data", False, transforms)
    if is_distribute:
        train_set = distributed_dataset(train_set, split, rank, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, drop_last=True)
    return train_loader, test_loader, (3, 64, 64), 200

