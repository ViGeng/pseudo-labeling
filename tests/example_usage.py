
import os
import sys
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.datasets.wrappers import ImageNetWrapper

def main():
    print("=== ImageNet Wrapper Usage Example ===\n")
    
    # 1. Configuration
    # The wrapper expects the standard ImageNet folder structure:
    # root/
    #   train/
    #     n01440764/
    #       img1.JPEG
    #       ...
    #   val/
    #     n01440764/
    #       img2.JPEG
    #       ...
    
    data_root = './data' 
    split = 'val'
    
    print(f"Config:")
    print(f"  Root:  {os.path.abspath(data_root)}")
    print(f"  Split: {split}")
    print("-" * 40)

    # 2. Initialization
    try:
        print(f"Initializing ImageNetWrapper...")
        # download=False because ImageNet takes ~150GB+ and cannot be auto-downloaded easily.
        # It relies on the user having the data in the expected structure.
        dataset = ImageNetWrapper(root=data_root, split=split, download=False)
        
        print(f"SUCCESS: Dataset initialized.")
        print(f"  Total Images: {len(dataset)}")
        
        # 3. Accessing Data
        if len(dataset) > 0:
            print("\nFetching first sample...")
            sample_id, img, target = dataset[0]
            
            print(f"  Sample ID:    {sample_id}")
            print(f"  Target Class: {target}")
            print(f"  Image Tensor: {img.shape} (Channels, Height, Width)")
            print(f"  Value Range:  [{img.min():.3f}, {img.max():.3f}] (Normalized)")
            
    except FileNotFoundError as e:
        print(f"\n[INFO] Dataset not found (Expected).")
        print(f"Reason: {e}")
        print("\nTo run this example with real data:")
        print(f"1. Create directory: {os.path.abspath(data_root)}/{split}")
        print("2. Add class folders (e.g., 'n01440764') containing images.")
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error:\n{e}")

    print("-" * 40)
    print("\n=== Streaming Dataset Wrapper Example ===\n")
    print("This wrapper allows iterating over datasets (hosted on Hugging Face) without full download.")
    print("It uses the 'datasets' library in streaming mode.")
    print("Requirement: 'datasets' library installed (pip install datasets).")
    
    try:
        from src.datasets.wrappers import StreamingImageNetWrapper
        
        # Using 'imagenet-1k' as requested.
        # Note: This requires huggingface-cli login and accepting terms.
        demo_dataset_name = 'imagenet-1k' 
        demo_split = 'validation'
        
        print(f"\nInitializing StreamingImageNetWrapper (dataset='{demo_dataset_name}', split='{demo_split}')...")
        print("Note: This fetches data on-the-fly from Hugging Face.")
        
        streaming_ds = StreamingImageNetWrapper(split='val', streaming=True, dataset_name=demo_dataset_name)
        
        print("Iterating to fetch first sample...")
        iterator = iter(streaming_ds)
        sample_id, img, target = next(iterator)
        
        print(f"SUCCESS: Fetched valid sample.")
        print(f"  Sample ID:    {sample_id}")
        print(f"  Target Class: {target}")
        print(f"  Image Tensor: {img.shape}")
        
    except ImportError:
        print("\n[SKIP] 'datasets' library not installed.")
        print("Run: pip install datasets")
    except Exception as e:
        print(f"\n[INFO] Streaming failed (likely due to Auth, Network, or compatibility):")
        print(f"  {e}")
        print("\nTo fix:")
        print("  1. pip install datasets huggingface_hub")
        print("  2. Check internet connection")
        print("  3. If using gated datasets (like imagenet-1k), run: huggingface-cli login")

    print("\n=== Done ===")

if __name__ == "__main__":
    main()
