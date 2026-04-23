from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from utils.config import ProjectConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_cifar10_dataloaders(config: ProjectConfig):
    seed_everything(config.experiment.seed)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(config.data.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config.data.mean, config.data.std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(config.data.mean, config.data.std),
        ]
    )

    data_dir = config.paths.resolve(config.paths.data_dir)
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    validation_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=eval_transform
    )
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)

    num_train = len(train_dataset)
    num_val = int(num_train * config.data.validation_split)
    indices = torch.randperm(num_train, generator=torch.Generator().manual_seed(config.experiment.seed)).tolist()
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]

    if config.data.train_subset_size is not None:
        train_indices = train_indices[: min(len(train_indices), config.data.train_subset_size)]
    if config.data.val_subset_size is not None:
        val_indices = val_indices[: min(len(val_indices), config.data.val_subset_size)]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(validation_dataset, val_indices)

    loader_kwargs = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory and torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_subset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_dataset.classes
