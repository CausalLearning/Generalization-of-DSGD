from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from torchvision.datasets import CIFAR100
from .distributed_dataset import distributed_dataset


def cifar100(rank, split=None, batch_size=None,
             transforms=None, test_batch_size=64,
             is_distribute=True, seed=777, **kwargs):
    if transforms is None:
        transforms = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
    if batch_size is None:
        batch_size = 1
    if split is None:
        split = [1.0]
    train_set = CIFAR100("../data", train=True, download=True, transform=transforms)
    test_set = CIFAR100("../data", train=False, download=True, transform=transforms)
    if is_distribute:
        train_set = distributed_dataset(train_set, split, rank, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size)
    return train_loader, test_loader, (3, 32, 32), 100
