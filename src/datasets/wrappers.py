import torch
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
import torchvision.transforms as T
from .base import BaseDataset
from typing import Tuple, Any

class CIFAR10Wrapper(BaseDataset):
    def __init__(self, root: str, split: str = 'val', download: bool = True):
        train = (split == 'train')
        self.dataset = CIFAR10(root=root, train=train, download=download, transform=T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]))
        self.split = split

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        img, target = self.dataset[index]
        # Generate a deterministic unique ID based on split and index
        sample_id = f"cifar10_{self.split}_{index}" 
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.dataset)

class CIFAR100Wrapper(BaseDataset):
    def __init__(self, root: str, split: str = 'val', download: bool = True):
        train = (split == 'train')
        self.dataset = CIFAR100(root=root, train=train, download=download, transform=T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]))
        self.split = split

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        img, target = self.dataset[index]
        sample_id = f"cifar100_{self.split}_{index}" 
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.dataset)

class FashionMNISTWrapper(BaseDataset):
    def __init__(self, root: str, split: str = 'val', download: bool = True):
        train = (split == 'train')
        self.dataset = FashionMNIST(root=root, train=train, download=download, transform=T.Compose([
            T.Grayscale(num_output_channels=3), # Expand to 3 channels for model compatibility
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ]))
        self.split = split

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        img, target = self.dataset[index]
        sample_id = f"fmnist_{self.split}_{index}" 
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.dataset)
