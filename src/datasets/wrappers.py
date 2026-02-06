import torch
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, Imagenette, ImageFolder
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

class ImagenetteWrapper(BaseDataset):
    def __init__(self, root: str, split: str = 'val', download: bool = True):
        # Imagenette 'split' argument is strictly 'train' or 'val'.
        # Our config might pass 'train' or 'test'. Map 'test' to 'val' if needed, or stick to standard.
        # torchvision.datasets.Imagenette uses split='train' or 'val'.
        self.split = 'train' if split == 'train' else 'val'
        
        self.dataset = Imagenette(root=root, split=self.split, download=download, transform=T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]))

        # Mapping from Imagenette class index (0-9) to ImageNet class index (0-999)
        # Order: tench, English springer, cassette player, chain saw, church, 
        # French horn, garbage truck, gas pump, golf ball, parachute
        self.imagenet_mapping = {
            0: 0,   # tench
            1: 217, # English springer
            2: 482, # cassette player
            3: 491, # chain saw
            4: 497, # church
            5: 566, # French horn
            6: 569, # garbage truck
            7: 571, # gas pump
            8: 574, # golf ball
            9: 701  # parachute
        }

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        img, target = self.dataset[index]
        
        # Map target to original ImageNet index
        target = self.imagenet_mapping.get(target, target)
        
        # Imagenette doesn't easily expose filenames in simple API (unlike ImageFolder), 
        # so we'll generate a deterministic ID.
        sample_id = f"imagenette_{self.split}_{index}"
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.dataset)

class ImageNetWrapper(BaseDataset):
    def __init__(self, root: str, split: str = 'val', download: bool = False):
        # ImageNet from torchvision is just ImageFolder usually, or the actual ImageNet class.
        # The actual ImageNet class is deprecated/removed in some versions or requires manual download.
        # We will use ImageFolder for maximum compatibility with user provided data folders.
        # Structure expected: root/train/... and root/val/...
        self.split = 'train' if split == 'train' else 'val'
        
        import os
        data_path = os.path.join(root, self.split)
        if not os.path.exists(data_path):
             # Fallback: maybe the root itself is the data folder?
             data_path = root
        
        self.dataset = ImageFolder(root=data_path, transform=T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]))

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        img, target = self.dataset[index]
        # ImageFolder samples are (path, class_index) tuples
        path, _ = self.dataset.samples[index]
        import os
        filename = os.path.basename(path)
        sample_id = filename
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.dataset)
