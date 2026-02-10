
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.append(os.getcwd())

# Mock dependencies before import
sys.modules['numpy'] = MagicMock()

# Mock torch and submodules
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch
mock_torch_utils = MagicMock()
sys.modules['torch.utils'] = mock_torch_utils
mock_torch_utils_data = MagicMock()
sys.modules['torch.utils.data'] = mock_torch_utils_data

# Ensure Dataset is a class we can inherit from
class MockDataset:
    pass
mock_torch_utils_data.Dataset = MockDataset

# Mock torchvision
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.datasets'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()

# Mock PIL
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()

# Mock datasets (HF) library
mock_datasets_lib = MagicMock()
sys.modules['datasets'] = mock_datasets_lib

# Mock internal downloader
# We need to mock src.utils.downloader BEFORE importing wrappers
import src.utils.downloader
mock_download_and_extract = MagicMock()
src.utils.downloader.download_and_extract = mock_download_and_extract

# Import wrappers
try:
    from src.datasets.wrappers import (
        CIFAR10Wrapper, CIFAR100Wrapper, FashionMNISTWrapper, ImagenetteWrapper, 
        StanfordCarsWrapper, ImageNetWrapper, ImageNetCWrapper, StylizedImageNetWrapper,
        OfficeHomeWrapper, DomainNetWrapper, CUB200Wrapper, CIFAR100CWrapper,
        TinyImageNetWrapper, TinyImageNetCWrapper
    )
except ImportError as e:
    print(f"Failed to import wrappers: {e}")
    sys.exit(1)

# Helper to mock torchvision dataset initialization
def mock_tv_init(self, root, train=True, transform=None, target_transform=None, download=False, **kwargs):
    self.root = root
    self.train = train
    self.transform = transform
    self.download = download
    # Check if download=True triggers something in real life, but here we just store it.
    # We can check if it was called with download=True.

class TestAllDownloads(unittest.TestCase):
    
    def setUp(self):
        # Reset mocks
        mock_download_and_extract.reset_mock()
        mock_datasets_lib.load_dataset.reset_mock()

    def test_torchvision_wrappers(self):
        """Test standard torchvision wrappers pass download=True to underlying dataset"""
        wrappers = [
            (CIFAR10Wrapper, 'CIFAR10'),
            (CIFAR100Wrapper, 'CIFAR100'),
            (FashionMNISTWrapper, 'FashionMNIST'),
            (StanfordCarsWrapper, 'StanfordCars'),
            # Imagenette uses a different underlying class often or custom logic, let's check wrapper.
            # Wrapper for Imagenette uses 'Imagenette' from torchvision (mocked)
            (ImagenetteWrapper, 'Imagenette'),
        ]
        
        for wrapper_cls, tv_name in wrappers:
            with self.subTest(dataset=tv_name):
                # We need to patch the class that the wrapper module USES.
                # Since we imported wrappers from src.datasets.wrappers, we should patch
                # src.datasets.wrappers.CIFAR10, etc.
                
                # However, wrappers.py does: from torchvision.datasets import CIFAR10, ...
                # So we need to patch 'src.datasets.wrappers.CIFAR10'
                
                target = f'src.datasets.wrappers.{tv_name}'
                # Special case for Imagenette if it's different mapping
                if tv_name == 'Imagenette':
                     # Imagenette might be a custom class or imported. 
                     # Checking wrappers.py: from torchvision.datasets import ..., Imagenette
                     pass

                with patch(target) as mock_ds_class:
                    # Instantiate
                    wrapper = wrapper_cls(root='./data', download=True)
                    
                    # Verify
                    self.assertTrue(mock_ds_class.called, f"{tv_name} not initialized")
                    _, kwargs = mock_ds_class.call_args
                    # Some datasets like Imagenette might wrap standard args differently, 
                    # but usually download is passed.
                    if tv_name == 'StanfordCars':
                         # StanfordCars args: root, split, download, transform
                         self.assertTrue(kwargs.get('download', False) or mock_ds_class.call_args[0][2]==True, "StanfordCars download arg missing")
                    elif tv_name == 'Imagenette':
                         # Imagenette: root, split, download, transform
                         self.assertTrue(kwargs.get('download', False), "Imagenette download=True missing")
                    else:
                         # CIFAR10/100/FashionMNIST: root, train, download, transform
                         self.assertTrue(kwargs.get('download', False), f"{tv_name} download=True missing")

    @patch('os.path.exists', return_value=False) # Force download
    def test_custom_downloaders(self, mock_exists):
        """Test wrappers that use src.utils.downloader.download_and_extract"""
        
        # 1. Tiny ImageNet
        # It checks train dir existence.
        # It calls download_and_extract with specific URL.
        try:
            # We need to mock os.listdir for TinyImageNet loading logic to not crash after download logic
            with patch('os.listdir', return_value=['n01234567']):
                 with patch('os.path.isdir', return_value=True):
                    wrapper = TinyImageNetWrapper(root='./data', download=True)
        except Exception:
            # It might fail later in _load_train, but we care if download was called
            pass
        
        self.assertTrue(mock_download_and_extract.called, "TinyImageNet did not attempt download")
        args, _ = mock_download_and_extract.call_args
        self.assertIn('tiny-imagenet-200.zip', args[0])
        mock_download_and_extract.reset_mock()

        # 2. CUB-200-2011
        try:
             # CUB reads files after download.
            with patch('builtins.open', MagicMock()):
                wrapper = CUB200Wrapper(root='./data/CUB_200_2011', download=True)
        except Exception:
            pass
            
        self.assertTrue(mock_download_and_extract.called, "CUB200 did not attempt download")
        args, _ = mock_download_and_extract.call_args
        self.assertIn('CUB_200_2011.tgz', args[0])
        mock_download_and_extract.reset_mock()

        # 3. DomainNet
        try:
            wrapper = DomainNetWrapper(root='./data', domain='real', download=True)
        except Exception:
            pass
        
        self.assertTrue(mock_download_and_extract.called, "DomainNet did not attempt download")
        args, _ = mock_download_and_extract.call_args
        # DomainNet 'real' url
        self.assertIn('real.zip', args[0])
        mock_download_and_extract.reset_mock()
        
        # 4. ImageNet-C
        try:
            wrapper = ImageNetCWrapper(root='./data', corruption_type='gaussian_noise', download=True)
        except Exception:
            pass
            
        self.assertTrue(mock_download_and_extract.called, "ImageNet-C did not attempt download")
        args, _ = mock_download_and_extract.call_args
        # gaussian_noise -> 'noise' category URL
        self.assertIn('noise.tar', args[0])
        mock_download_and_extract.reset_mock()
        
        # 5. Tiny-ImageNet-C
        try:
            wrapper = TinyImageNetCWrapper(root='./data', corruption_type='gaussian_noise', download=True)
        except Exception:
            pass
            
        self.assertTrue(mock_download_and_extract.called, "Tiny-ImageNet-C did not attempt download")
        args, _ = mock_download_and_extract.call_args
        self.assertIn('Tiny-ImageNet-C.tar', args[0])
        mock_download_and_extract.reset_mock()
        
        # 6. CIFAR-100-C
        try:
            # Needs npy load
            with patch('numpy.load', return_value=MagicMock()):
                wrapper = CIFAR100CWrapper(root='./data', corruption_type='gaussian_noise', download=True)
        except Exception:
            pass
            
        self.assertTrue(mock_download_and_extract.called, "CIFAR-100-C did not attempt download")
        args, _ = mock_download_and_extract.call_args
        self.assertIn('CIFAR-100-C.tar', args[0])
        mock_download_and_extract.reset_mock()

    @patch('os.path.exists', return_value=False)
    def test_office_home(self, mock_exists):
        """Test OfficeHome uses datasets library"""
        # Prepare HF Mock
        mock_ds = MagicMock()
        mock_datasets_lib.load_dataset.return_value = mock_ds
        mock_ds.filter.return_value = MagicMock()
        
        wrapper = OfficeHomeWrapper(root='./data', download=True)
        
        mock_datasets_lib.load_dataset.assert_called_with("flwrlabs/office-home", split="train")

    def test_manual_download_datasets(self):
        """Test datasets that do NOT support auto-download raise error or instruction"""
        # ImageNet
        with patch('os.path.exists', return_value=False):
            # It should just default to ImageFolder and probably fail if path doesn't exist, 
            # OR logic in wrapper might not check download flag if it's False by default.
            # ImageNetWrapper in wrappers.py: download: bool = False.
            # If we force True, does it try to download? The code says:
            # def __init__(..., download=False): ... data_path = ...; self.dataset = ImageFolder...
            # It seems it doesn't implement auto-download logic.
            # Let's verify it doesn't try to call download_and_extract
            
            try:
                # Mock ImageFolder to avoid FileNotFoundError if it tries to list dirs
                with patch('torchvision.datasets.ImageFolder'):
                    wrapper = ImageNetWrapper(root='./data', download=True)
            except Exception:
                pass
            
            mock_download_and_extract.assert_not_called()
            
        # StylizedImageNet
        with patch('os.path.exists', return_value=False):
            try:
                wrapper = StylizedImageNetWrapper(root='./data', download=True)
            except FileNotFoundError as e:
                # It should raise FileNotFoundError with instructions
                self.assertIn("Download from", str(e))
                
            mock_download_and_extract.assert_not_called()

if __name__ == '__main__':
    unittest.main(verbosity=2)
