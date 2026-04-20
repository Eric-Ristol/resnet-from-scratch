#CIFAR-10 data pipeline.
#CIFAR-10 has 60,000 32x32 colour images in 10 classes:
#  airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
#Standard split is 50,000 train / 10,000 test.
#Training-time augmentation (same recipe as the original ResNet paper
#for CIFAR): pad the 32x32 image with 4 pixels of zeros, random-crop
#back to 32x32, and random horizontal flip. Eval-time: no augmentation.
#Normalisation uses the per-channel mean/std of CIFAR-10's training set.
#These numbers are standard across every ResNet-on-CIFAR codebase;

import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

#CIFAR-10 training set statistics. Widely-used canonical values.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

#Where torchvision will cache the raw dataset. Kept inside the project so
#nothing leaks into the user's ~/.cache.
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _train_transform():
    #Augmentation + normalisation for the training split.
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _eval_transform():
    #Deterministic preprocessing for validation / test.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_datasets(download=True):
    #Returns (train_dataset, test_dataset) as torchvision CIFAR10 objects.
    os.makedirs(DATA_DIR, exist_ok=True)
    train = datasets.CIFAR10(
        root=DATA_DIR, train=True,  download=download, transform=_train_transform()
    )
    test  = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=download, transform=_eval_transform()
    )
    return train, test


def get_loaders(batch_size=128, num_workers=2, download=True):
    #Returns (train_loader, test_loader). Workers default to 2 -- the
    #Trainer-style sweet spot for CIFAR on a laptop. Set to 0 on very
    #constrained machines.
    train_ds, test_ds = get_datasets(download=download)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, test_loader


def get_tiny_subset_loaders(n_train=256, n_test=128, batch_size=64,
                            num_workers=0, download=True):
    #A very small subset used by unit tests and smoke-training runs.
    #Lets us verify the training loop end-to-end in a few seconds without
    #needing the full 50k-image pass.
    train_ds, test_ds = get_datasets(download=download)
    train_subset = Subset(train_ds, list(range(min(n_train, len(train_ds)))))
    test_subset  = Subset(test_ds,  list(range(min(n_test,  len(test_ds)))))

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, test_loader


def denormalize(tensor):
    #Utility: undo the Normalize step, useful for plotting.
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return tensor * std + mean
