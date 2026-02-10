
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
# Assuming we run from project root: python3 scripts/verify_officehome_logic.py
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

# Import the wrapper to test
# We use src.datasets.wrappers because src is a package (namespace or simple folder in root)
try:
    from src.datasets.wrappers import OfficeHomeWrapper
except ImportError as e:
    print(f"Failed to import OfficeHomeWrapper: {e}")
    sys.exit(1)

class TestOfficeHomeWrapper(unittest.TestCase):
    def test_mock_import(self):
        """Test that we can import the wrapper"""
        self.assertIsNotNone(OfficeHomeWrapper)

    def test_auto_download_logic(self):
        """Test auto-download logic triggers HF datasets loading"""
        
        # Prepare mock for 'datasets' library
        mock_datasets_lib = MagicMock()
        mock_load_dataset = MagicMock()
        mock_datasets_lib.load_dataset = mock_load_dataset
        
        # Mock returned dataset
        mock_ds_dict = MagicMock()
        mock_filtered_ds = MagicMock()
        mock_filtered_ds.__len__.return_value = 10
        mock_filtered_ds.__getitem__.return_value = {
            'image': MagicMock(mode='RGB'),
            'label': 0,
            'domain': 'Art'
        }
        
        # load_dataset return value
        mock_load_dataset.return_value = mock_ds_dict
        mock_ds_dict.filter.return_value = mock_filtered_ds
        
        # Patch sys.modules to inject our mock 'datasets' lib
        with patch.dict(sys.modules, {'datasets': mock_datasets_lib}):
            # Patch os.path.exists to force download path (return False)
            with patch('os.path.exists', return_value=False):
                # Instantiate wrapper
                wrapper = OfficeHomeWrapper(root='./data', download=True, domain='Art')
                
                # Check load_dataset called
                mock_load_dataset.assert_called_with("flwrlabs/office-home", split="train")
                
                # Check filter called
                mock_ds_dict.filter.assert_called()
                
                # Check len
                self.assertEqual(len(wrapper), 10)
                
                # Check getitem
                sid, img, target = wrapper[0]
                self.assertTrue(sid.startswith("office_home_Art_"))
                self.assertEqual(target, 0)
                
                print("Verification successful: load_dataset called and data retrieved.")

if __name__ == '__main__':
    unittest.main()
