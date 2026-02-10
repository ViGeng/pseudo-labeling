import os
import sys
import torch
from torch.utils.data import DataLoader
from hydra import initialize, compose
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.wrappers import StreamingImageNetWrapper

def verify_streaming():
    print("Initializing StreamingImageNetWrapper...")
    # Default uses 'imagenet-1k' (requires HF auth). 
    # To test without auth, use: dataset = StreamingImageNetWrapper(split='val', streaming=True, dataset_name="cifar10")
    try:
        dataset = StreamingImageNetWrapper(split='val', streaming=True) # Defaults to imagenet-1k
    except Exception as e:
        print(f"Failed to load ImageNet (likely due to missing HF auth): {e}")
        print("Falling back to CIFAR-10 for logic verification...")
        dataset = StreamingImageNetWrapper(split='val', streaming=True, dataset_name="cifar10")
    
    # Test __iter__ via DataLoader
    loader = DataLoader(dataset, batch_size=2)
    
    print("Iterating over dataset...")
    for i, batch in enumerate(loader):
        sample_ids, images, targets = batch
        print(f"Batch {i}:")
        print(f"  Sample IDs: {sample_ids}")
        print(f"  Images shape: {images.shape}")
        print(f"  Targets: {targets}")
        
        if i >= 1: # Test 2 batches
            break
            
    print("Verification successful!")

if __name__ == "__main__":
    verify_streaming()
