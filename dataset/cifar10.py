from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from torchvision.datasets import CIFAR10
from .distributed_dataset import distributed_dataset


def cifar10(rank, split=None, batch_size=None,
            transforms=None, test_batch_size=64,
            is_distribute=True, seed=777, **kwargs):
    if transforms is None:
        transforms = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize([0.4940607, 0.4850613, 0.45037037], [0.20085774, 0.19870903, 0.20153421])
        ])
    if batch_size is None:
        batch_size = 1
    if split is None:
        split = [1.0]
    train_set = CIFAR10("../data", train=True, download=True, transform=transforms)
    test_set = CIFAR10("../data", train=False, download=True, transform=transforms)
    if is_distribute:
        train_set = distributed_dataset(train_set, split, rank, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, drop_last=True)
    return train_loader, test_loader, (3, 32, 32), 10
