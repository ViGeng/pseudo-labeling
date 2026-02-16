import os
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision.datasets import (CIFAR10, CIFAR100, FashionMNIST, ImageFolder,
                                  Imagenette, StanfordCars)

from ..utils.downloader import download_and_extract
from .base import BaseDataset

# =============================================================================
# Standard ImageNet Transforms
# =============================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_imagenet_transform(resize=256, crop=224):
    """Standard ImageNet preprocessing transform."""
    return T.Compose([
        T.Resize(resize),
        T.CenterCrop(crop),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


# =============================================================================
# CIFAR-10 Wrapper
# Labels: 10 | Scale: 60K | Domain: Tiny natural images (32x32)
# =============================================================================
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
        sample_id = f"cifar10_{self.split}_{index}" 
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.dataset)


# =============================================================================
# CIFAR-100 Wrapper
# Labels: 100 | Scale: 60K | Domain: Tiny natural images (32x32)
# =============================================================================
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


# =============================================================================
# FashionMNIST Wrapper
# Labels: 10 | Scale: 70K | Domain: Grayscale fashion items (28x28)
# =============================================================================
class FashionMNISTWrapper(BaseDataset):
    def __init__(self, root: str, split: str = 'val', download: bool = True):
        train = (split == 'train')
        self.dataset = FashionMNIST(root=root, train=train, download=download, transform=T.Compose([
            T.Grayscale(num_output_channels=3),  # Expand to 3 channels for model compatibility
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


# =============================================================================
# Imagenette Wrapper
# Labels: 10 | Scale: ~13K | Domain: Easy ImageNet subset (10 classes)
# =============================================================================
class ImagenetteWrapper(BaseDataset):
    def __init__(self, root: str, split: str = 'val', download: bool = True):
        self.split = 'train' if split == 'train' else 'val'
        self.dataset = Imagenette(root=root, split=self.split, download=download, 
                                  transform=get_imagenet_transform())

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        img, target = self.dataset[index]
        sample_id = f"imagenette_{self.split}_{index}"
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.dataset)


# =============================================================================
# ImageNet Wrapper (using ImageFolder)
# Labels: 1000 | Scale: ~1.2M | Domain: Natural images (224x224)
# =============================================================================
class ImageNetWrapper(BaseDataset):
    def __init__(self, root: str, split: str = 'val', download: bool = False):
        self.split = 'train' if split == 'train' else 'val'
        data_path = os.path.join(root, self.split)
        if not os.path.exists(data_path):
            data_path = root
        self.dataset = ImageFolder(root=data_path, transform=get_imagenet_transform())

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        img, target = self.dataset[index]
        path, _ = self.dataset.samples[index]
        sample_id = os.path.basename(path)
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.dataset)


# =============================================================================
# ImageNet-C Wrapper (Corrupted ImageNet)
# Labels: 1000 | Scale: ~50K per corruption | Domain: Corrupted natural images
# Use: Texture vs Shape bias testing
# =============================================================================
class ImageNetCWrapper(BaseDataset):
    """
    ImageNet-C: corrupted version of ImageNet validation set.
    Structure: root/<corruption_type>/<severity>/<class>/<images>

    Auto-download: Downloads per-category tarballs from Zenodo (~7-23GB each).
    Categories: noise, blur, weather, digital, extra.
    """
    # Corruption type -> Zenodo tarball mapping
    _CORRUPTION_CATEGORY = {
        'gaussian_noise': 'noise', 'shot_noise': 'noise', 'impulse_noise': 'noise',
        'defocus_blur': 'blur', 'glass_blur': 'blur', 'motion_blur': 'blur', 'zoom_blur': 'blur',
        'snow': 'weather', 'frost': 'weather', 'fog': 'weather', 'brightness': 'weather',
        'contrast': 'digital', 'elastic_transform': 'digital',
        'pixelate': 'digital', 'jpeg_compression': 'digital',
    }
    _ZENODO_URLS = {
        'noise': 'https://zenodo.org/records/2235448/files/noise.tar',
        'blur': 'https://zenodo.org/records/2235448/files/blur.tar',
        'weather': 'https://zenodo.org/records/2235448/files/weather.tar',
        'digital': 'https://zenodo.org/records/2235448/files/digital.tar',
    }

    def __init__(self, root: str, split: str = 'val', download: bool = True,
                 corruption_type: str = 'gaussian_noise', severity: int = 3):
        self.split = split
        self.corruption_type = corruption_type
        self.severity = severity
        
        # Build path: root/<corruption_type>/<severity>
        data_path = os.path.join(root, corruption_type, str(severity))
        if not os.path.exists(data_path) and download:
            self._auto_download(root, corruption_type)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"ImageNet-C data not found at {data_path}. "
                f"Download from https://zenodo.org/records/2235448"
            )
            
        self.dataset = ImageFolder(root=data_path, transform=get_imagenet_transform())

    @classmethod
    def _auto_download(cls, root, corruption_type):
        category = cls._CORRUPTION_CATEGORY.get(corruption_type)
        if category and category in cls._ZENODO_URLS:
            print(f"Auto-downloading ImageNet-C '{category}' category (may be large)...")
            download_and_extract(cls._ZENODO_URLS[category], root)
        else:
            print(f"No auto-download URL for corruption '{corruption_type}'.")
        
        self.dataset = ImageFolder(root=data_path, transform=get_imagenet_transform())

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        img, target = self.dataset[index]
        path, _ = self.dataset.samples[index]
        sample_id = f"imagenet_c_{self.corruption_type}_{self.severity}_{os.path.basename(path)}"
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.dataset)


# =============================================================================
# Stylized-ImageNet Wrapper
# Labels: 1000 | Scale: ~1.2M | Domain: Style-transferred natural images
# Use: Texture vs Shape bias testing (texture removed, shape preserved)
# =============================================================================
class StylizedImageNetWrapper(BaseDataset):
    """
    Stylized-ImageNet: ImageNet with style transfer applied.
    Structure: root/<split>/<class>/<images>
    """
    def __init__(self, root: str, split: str = 'val', download: bool = False):
        self.split = 'train' if split == 'train' else 'val'
        data_path = os.path.join(root, self.split)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Stylized-ImageNet data not found at {data_path}. "
                f"Download from https://github.com/rgeirhos/Stylized-ImageNet"
            )
        self.dataset = ImageFolder(root=data_path, transform=get_imagenet_transform())

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        img, target = self.dataset[index]
        path, _ = self.dataset.samples[index]
        sample_id = f"sin_{self.split}_{os.path.basename(path)}"
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.dataset)


# =============================================================================
# Office-Home Wrapper
# Labels: 65 | Scale: ~15.5K | Domains: Art, Clipart, Product, Real_World
# Use: Specialist vs Generalist gap testing
# =============================================================================
class OfficeHomeWrapper(BaseDataset):
    """
    Office-Home: Multi-domain dataset for domain adaptation.
    Structure: root/<domain>/<class>/<images>
    
    Auto-download: Downloads from Hugging Face (flwrlabs/office-home).
    """
    DOMAINS = ['Art', 'Clipart', 'Product', 'Real_World']
    
    def __init__(self, root: str, split: str = 'val', download: bool = False,
                 domain: str = 'Real_World'):
        self.split = split
        self.domain = domain
        self.root = root
        
        if domain not in self.DOMAINS:
            raise ValueError(f"Domain must be one of {self.DOMAINS}, got {domain}")
        
        # Check for local data first
        data_path = os.path.join(root, domain)
        self.use_hf_dataset = False
        
        if os.path.exists(data_path):
            print(f"Loading Office-Home '{domain}' from local disk: {data_path}")
            self.dataset = ImageFolder(root=data_path, transform=get_imagenet_transform())
        elif download:
            print(f"Local data not found at {data_path}. Attempting download via Hugging Face datasets...")
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError("The 'datasets' library is required for auto-download. Please install it via 'pip install datasets'.")
            
            # flwrlabs/office-home has a 'train' split. We filter by domain.
            # Note: The HF dataset has columns: image, label, domain
            print(f"Downloading/Loading flwrlabs/office-home config='{domain}'...")
            
            # The dataset on HF (flwrlabs/office-home) might not have 'domain' config, but 'domain' column.
            # Based on user info: load_dataset("flwrlabs/office-home")
            # We load the whole thing (it's small, ~2-3GB total) or stream? 
            # 3GB is fine to download. 
            self.hf_dataset = load_dataset("flwrlabs/office-home", split="train")
            
            # Filter by domain
            print(f"Filtering for domain '{domain}'...")
            self.dataset = self.hf_dataset.filter(lambda x: x['domain'] == domain)
            self.use_hf_dataset = True
            
            self.transform = get_imagenet_transform()
        else:
            raise FileNotFoundError(
                f"Office-Home data not found at {data_path}. "
                f"Set download=True to auto-download, or manually download from https://www.hemanthdv.org/officeHomeDataset.html"
            )

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        if self.use_hf_dataset:
            item = self.dataset[index]
            img = item['image']
            target = item['label']
            
            # HF dataset might not have filenames. We construct a unique ID.
            # If 'image_file_path' or similar exists, use it. Otherwise use index.
            # We check if we can get a filename, but for now defaulting to index-based ID.
            # Construct a consistent ID: office_home_{domain}_{index}
            # Note: This index is relative to the filtered dataset.
            sample_id = f"office_home_{self.domain}_{index}"
            
            if self.transform:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = self.transform(img)
                
            return sample_id, img, target
        else:
            # ImageFolder behavior
            img, target = self.dataset[index]
            path, _ = self.dataset.samples[index]
            sample_id = f"office_home_{self.domain}_{os.path.basename(path)}"
            return sample_id, img, target

    def __len__(self) -> int:
        return len(self.dataset)


# =============================================================================
# DomainNet Wrapper
# Labels: 345 | Scale: ~600K | Domains: clipart, infograph, painting, quickdraw, real, sketch
# Use: Specialist vs Generalist gap testing (large scale)
# =============================================================================
class DomainNetWrapper(BaseDataset):
    """
    DomainNet: Large-scale multi-domain dataset.
    Structure: root/<domain>/<class>/<images>

    Auto-download: Downloads per-domain zips from BU (~0.4-5.6GB each).
    """
    DOMAINS = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    _DOMAIN_URLS = {
        'clipart': 'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip',
        'infograph': 'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip',
        'painting': 'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip',
        'quickdraw': 'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
        'real': 'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
        'sketch': 'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip',
    }
    
    def __init__(self, root: str, split: str = 'test', download: bool = True,
                 domain: str = 'real'):
        self.split = split
        self.domain = domain
        
        if domain not in self.DOMAINS:
            raise ValueError(f"Domain must be one of {self.DOMAINS}, got {domain}")
        
        data_path = os.path.join(root, domain)
        if not os.path.exists(data_path) and download:
            url = self._DOMAIN_URLS[domain]
            print(f"Auto-downloading DomainNet '{domain}' domain...")
            download_and_extract(url, root)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"DomainNet data not found at {data_path}. "
                f"Download from http://ai.bu.edu/M3SDA/"
            )
        
        self.dataset = ImageFolder(root=data_path, transform=get_imagenet_transform())

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        img, target = self.dataset[index]
        path, _ = self.dataset.samples[index]
        sample_id = f"domainnet_{self.domain}_{os.path.basename(path)}"
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.dataset)


# =============================================================================
# Stanford Cars Wrapper
# Labels: 196 | Scale: ~16K | Domain: Fine-grained car images
# Use: Resolution & Crop inversion testing
# =============================================================================
class StanfordCarsWrapper(BaseDataset):
    """
    Stanford Cars: Fine-grained car classification.
    Uses torchvision.datasets.StanfordCars with auto-download support.
    """
    def __init__(self, root: str, split: str = 'test', download: bool = True):
        self.split = 'train' if split == 'train' else 'test'
        self.dataset = StanfordCars(
            root=root, 
            split=self.split, 
            download=download,
            transform=get_imagenet_transform()
        )

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        img, target = self.dataset[index]
        sample_id = f"stanford_cars_{self.split}_{index}"
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.dataset)


# =============================================================================
# CUB-200-2011 Wrapper (Caltech-UCSD Birds)
# Labels: 200 | Scale: ~12K | Domain: Fine-grained bird images
# Use: Resolution & Crop inversion testing
# =============================================================================
class CUB200Wrapper(BaseDataset):
    """
    CUB-200-2011: Fine-grained bird species classification.
    Structure: root/images/<class>/<images>

    Auto-download: Downloads from CaltechDATA (~1.2GB).
    """
    _DOWNLOAD_URL = 'https://data.caltech.edu/records/20098/files/CUB_200_2011.tgz'

    def __init__(self, root: str, split: str = 'test', download: bool = True):
        self.split = split
        self.root = root
        
        # Read train/test split file
        split_file = os.path.join(root, 'train_test_split.txt')
        images_file = os.path.join(root, 'images.txt')
        labels_file = os.path.join(root, 'image_class_labels.txt')
        
        if not all(os.path.exists(f) for f in [split_file, images_file, labels_file]):
            if download:
                # Download and extract; tgz extracts to CUB_200_2011/ inside parent dir
                parent_dir = os.path.dirname(root)
                download_and_extract(self._DOWNLOAD_URL, parent_dir)
            if not all(os.path.exists(f) for f in [split_file, images_file, labels_file]):
                raise FileNotFoundError(
                    f"CUB-200-2011 data not found at {root}. "
                    f"Download from https://www.vision.caltech.edu/datasets/cub_200_2011/"
                )
        
        # Parse split info (1=train, 0=test)
        with open(split_file, 'r') as f:
            splits = {int(line.split()[0]): int(line.split()[1]) for line in f}
        
        # Parse image paths
        with open(images_file, 'r') as f:
            images = {int(line.split()[0]): line.split()[1] for line in f}
        
        # Parse labels (1-indexed, convert to 0-indexed)
        with open(labels_file, 'r') as f:
            labels = {int(line.split()[0]): int(line.split()[1]) - 1 for line in f}
        
        # Filter by split
        is_train = 1 if split == 'train' else 0
        self.samples = []
        for img_id, img_path in images.items():
            if splits[img_id] == is_train:
                full_path = os.path.join(root, 'images', img_path)
                self.samples.append((full_path, labels[img_id]))
        
        self.transform = get_imagenet_transform()

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        sample_id = f"cub200_{self.split}_{os.path.basename(path)}"
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.samples)


# =============================================================================
# CIFAR-100-C Wrapper (Corrupted CIFAR-100)
# Labels: 100 | Scale: ~10K per corruption | Domain: Corrupted tiny images
# Use: Robustness Trap (AugMix) testing
# =============================================================================
class CIFAR100CWrapper(BaseDataset):
    """
    CIFAR-100-C: corrupted version of CIFAR-100 test set.
    Structure: root/<corruption_type>.npy and root/labels.npy

    Auto-download: Downloads from Zenodo (~2.9GB).
    """
    _DOWNLOAD_URL = 'https://zenodo.org/records/3555552/files/CIFAR-100-C.tar'

    CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur',
        'snow', 'frost', 'fog', 'brightness', 'spatter',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'saturate'
    ]
    
    def __init__(self, root: str, split: str = 'test', download: bool = True,
                 corruption_type: str = 'gaussian_noise', severity: int = 3):
        self.split = split
        self.corruption_type = corruption_type
        self.severity = severity
        
        if corruption_type not in self.CORRUPTIONS:
            raise ValueError(f"Corruption must be one of {self.CORRUPTIONS}")
        if not 1 <= severity <= 5:
            raise ValueError("Severity must be between 1 and 5")
        
        # Load corruption data (shape: 50000, 32, 32, 3) for all 5 severities
        corruption_file = os.path.join(root, f'{corruption_type}.npy')
        labels_file = os.path.join(root, 'labels.npy')
        
        if not os.path.exists(corruption_file) and download:
            # tar extracts to CIFAR-100-C/ inside parent dir
            parent_dir = os.path.dirname(root)
            download_and_extract(self._DOWNLOAD_URL, parent_dir)
        if not os.path.exists(corruption_file):
            raise FileNotFoundError(
                f"CIFAR-100-C data not found at {corruption_file}. "
                f"Download from https://zenodo.org/records/3555552"
            )
        
        # Each corruption has 10000 images per severity level (50000 total)
        all_images = np.load(corruption_file)
        all_labels = np.load(labels_file)
        
        # Select severity level (0-indexed: severity 1 = [0:10000], severity 2 = [10000:20000], etc.)
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        
        self.images = all_images[start_idx:end_idx]
        self.labels = all_labels[start_idx:end_idx]
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        img = Image.fromarray(self.images[index])
        target = int(self.labels[index])
        img = self.transform(img)
        sample_id = f"cifar100c_{self.corruption_type}_{self.severity}_{index}"
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.images)


# =============================================================================
# Tiny ImageNet Wrapper
# Labels: 200 | Scale: 110K | Domain: Resized natural images (64x64)
# =============================================================================
class TinyImageNetWrapper(BaseDataset):
    """
    Tiny ImageNet: 200-class subset of ImageNet at 64x64 resolution.
    Structure: root/train/<class>/images/... and root/val/images/...

    Auto-download: Downloads from Stanford CS231N (~237MB).
    """
    _DOWNLOAD_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    def __init__(self, root: str, split: str = 'val', download: bool = True):
        self.split = split
        self.root = root
        
        # Auto-download if not present
        train_dir = os.path.join(root, 'train')
        if not os.path.exists(train_dir) and download:
            parent_dir = os.path.dirname(root)
            download_and_extract(self._DOWNLOAD_URL, parent_dir)
        
        if split == 'train':
            self._load_train()
        else:
            self._load_val()
        
        self.transform = T.Compose([
            T.Resize(64),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    
    def _load_train(self):
        train_dir = os.path.join(self.root, 'train')
        if not os.path.exists(train_dir):
            raise FileNotFoundError(
                f"Tiny ImageNet train data not found at {train_dir}. "
                f"Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            )
        
        # Build class to index mapping
        classes = sorted(os.listdir(train_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        self.samples = []
        for cls in classes:
            cls_dir = os.path.join(train_dir, cls, 'images')
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    if img_name.endswith('.JPEG'):
                        self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))
    
    def _load_val(self):
        val_dir = os.path.join(self.root, 'val')
        annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(
                f"Tiny ImageNet val annotations not found at {annotations_file}. "
                f"Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            )
        
        # Build class to index mapping from train folder
        train_dir = os.path.join(self.root, 'train')
        classes = sorted(os.listdir(train_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Parse annotations
        self.samples = []
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                cls = parts[1]
                img_path = os.path.join(val_dir, 'images', img_name)
                self.samples.append((img_path, self.class_to_idx[cls]))

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        sample_id = f"tiny_imagenet_{self.split}_{os.path.basename(path)}"
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.samples)


# =============================================================================
# Tiny-ImageNet-C Wrapper (Corrupted Tiny ImageNet)
# Labels: 200 | Scale: ~10K per corruption | Domain: Corrupted small images
# Use: Robustness Trap (AugMix) testing at medium scale
# =============================================================================
class TinyImageNetCWrapper(BaseDataset):
    """
    Tiny-ImageNet-C: corrupted version of Tiny ImageNet.
    Structure: root/<corruption>/<severity>/<class>/<images>

    Auto-download: Downloads from Zenodo (~1.8GB).
    """
    _DOWNLOAD_URL = 'https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar'

    def __init__(self, root: str, split: str = 'val', download: bool = True,
                 corruption_type: str = 'gaussian_noise', severity: int = 3):
        self.split = split
        self.corruption_type = corruption_type
        self.severity = severity
        
        # Build path: root/<corruption>/<severity>
        data_path = os.path.join(root, corruption_type, str(severity))
        if not os.path.exists(data_path) and download:
            parent_dir = os.path.dirname(root)
            download_and_extract(self._DOWNLOAD_URL, parent_dir)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Tiny-ImageNet-C data not found at {data_path}. "
                f"Download from https://zenodo.org/records/2536630"
            )
        
        self.dataset = ImageFolder(root=data_path, transform=T.Compose([
            T.Resize(64),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]))

    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        img, target = self.dataset[index]
        path, _ = self.dataset.samples[index]
        sample_id = f"tiny_imagenet_c_{self.corruption_type}_{self.severity}_{os.path.basename(path)}"
        return sample_id, img, target

    def __len__(self) -> int:
        return len(self.dataset)


# =============================================================================
# Streaming ImageNet Wrapper
# Labels: 1000 | Scale: ~1.2M | Domain: Natural images (224x224)
# Use: Large scale training/testing without full download
# =============================================================================
class StreamingImageNetWrapper(IterableDataset):
    """
    Streaming ImageNet: Uses Hugging Face 'datasets' streaming mode.
    Structure: Streaming access to imagenet-1k.
    """
    def __init__(self, split: str = 'val', streaming: bool = True, dataset_name: str = "imagenet-1k"):
        self.split = 'train' if split == 'train' else 'validation' # HF uses 'validation'
        if dataset_name == 'cifar10' and self.split == 'validation': self.split = 'test' # HF cifar10 uses 'test'

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("The 'datasets' library is required. Please install it via 'pip install datasets'.")
            
        print(f"Loading {dataset_name} (streaming={streaming}) split='{self.split}'...")
        self.dataset = load_dataset(dataset_name, split=self.split, streaming=streaming)
        self.transform = get_imagenet_transform()

    def __iter__(self):
        for i, item in enumerate(self.dataset):
            if 'image' in item:
                img = item['image']
            elif 'img' in item:
                img = item['img']
            else:
                raise KeyError(f"Could not find 'image' or 'img' in dataset item. Keys: {item.keys()}")
            
            target = item['label']
            
            # Use enumeration index for guaranteed uniqueness in streaming
            sample_id = f"imagenet_streaming_{self.split}_{i}" 

            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = self.transform(img)
            yield sample_id, img, target

