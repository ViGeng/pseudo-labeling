import sys
import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from training.finetuner import FineTuner

def test_finetuner():
    print("--- Testing FineTuner ---")
    
    # 1. Setup Mock Data
    inputs = torch.randn(10, 3, 32, 32)
    targets = torch.randint(0, 10, (10,))
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=5)
    
    # 2. Setup Model (ResNet18) - ImageNet usually has 1000 classes
    model = models.resnet18(weights=None) # Random weights for speed
    print(f"Original final layer: {model.fc}")
    
    # 3. Load Config
    config = {
        "epochs": 1,
        "lr": 0.01,
        "device": "cpu", # Use CPU for quick test
        "freeze_backbone": True
    }
    
    finetuner = FineTuner(config)
    
    # 4. Test Skip Logic
    print("\n[Test 1] Skip Logic (imagenet -> imagenet)")
    model_skipped = finetuner.tune(model, loader, source_dataset="imagenet", target_dataset="imagenet")
    assert model_skipped is model, "Model should be returned as-is"
    assert model.fc.out_features == 1000, "Head should not change"
    print("PASS: Skip logic works.")

    # 5. Test Fine-tuning (imagenet -> cifar10)
    print("\n[Test 2] Fine-tuning (imagenet -> cifar10)")
    model_tuned = finetuner.tune(model, loader, source_dataset="imagenet", target_dataset="cifar10", num_classes=10)
    
    assert model_tuned.fc.out_features == 10, f"Head should be 10, got {model_tuned.fc.out_features}"
    assert model_tuned.fc.weight.requires_grad == True, "Head should be trainable"
    # Check backbone frozen
    assert model_tuned.layer1[0].conv1.weight.requires_grad == False, "Backbone should be frozen"
    
    print("PASS: Head replacement and freezing works.")
    print("Fine-tuning ran successfully (no errors during training loop).")

if __name__ == "__main__":
    test_finetuner()
