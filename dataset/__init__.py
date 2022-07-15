from .distributed_dataset import DistributedDataset, distributed_dataset
from .cifar100 import cifar100
from .cifar10 import cifar10
from .tiny_imagenet import tiny_imagenet


def get_dataset(rank, dataset_name,
                split=None, batch_size=None,
                transforms=None, is_distribute=True,
                seed=777, **kwargs):
    if dataset_name == "CIFAR10":
        return cifar10(rank=rank,
                       split=split,
                       batch_size=batch_size,
                       transforms=transforms,
                       is_distribute=is_distribute,
                       seed=seed,
                       **kwargs)
    elif dataset_name == "CIFAR100":
        return cifar100(rank=rank,
                        split=split,
                        batch_size=batch_size,
                        transforms=transforms,
                        is_distribute=is_distribute,
                        seed=seed,
                        **kwargs)
    elif dataset_name == "TinyImageNet":
        return tiny_imagenet(rank=rank,
                             split=split,
                             batch_size=batch_size,
                             transforms=transforms,
                             is_distribute=is_distribute,
                             seed=seed,
                             **kwargs)
