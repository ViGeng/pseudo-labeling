
import json
import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import resnet18, ResNet18_Weights

def generate_mappings():
    # Load ImageNet classes
    weights = ResNet18_Weights.DEFAULT
    imagenet_classes = weights.meta["categories"]
    
    # CIFAR10 classes
    cifar10_classes = CIFAR10(root='./data', train=True, download=True).classes
    print(f"CIFAR10 Classes: {cifar10_classes}")
    
    # CIFAR100 classes
    cifar100_classes = CIFAR100(root='./data', train=True, download=True).classes
    print(f"CIFAR100 Classes: {cifar100_classes}")
    
    # 1. ImageNet -> CIFAR10
    # Map logic: check if imagenet class name contains cifar10 class name or vice versa
    # This is heuristic.
    
    # Custom synonyms/supercategories for CIFAR10
    cifar10_synonyms = {
        'airplane': ['airliner', 'warplane', 'wing'],
        'automobile': ['car', 'minivan', 'jeep', 'cab', 'taxi', 'wagon', 'convertible', 'roadster', 'coupe', 'limousine'],
        'bird': ['bird', 'ostrich', 'brambling', 'goldfinch', 'junco', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel', 'kite', 'bald eagle', 'vulture', 'owl', 'salamander', 'partridge'], # Added common birds manually if needed, but string search helps
        'cat': ['cat', 'tabby', 'tiger cat', 'persian cat', 'siamese cat', 'egyptian cat'],
        'deer': ['deer', 'gazelle', 'impala'],
        'dog': ['dog'], # ImageNet has many dog breeds, usually containing "terrier", "spaniel", "retriever", "dog", etc.
        'frog': ['frog', 'bullfrog', 'tree frog', 'tailed frog'],
        'horse': ['horse', 'sorrel', 'zebra'],
        'ship': ['ship', 'boat', 'liner', 'carrier', 'submarine'],
        'truck': ['truck', 'pickup', 'trailer', 'van']
    }
    
    mapping_c10 = {}
    
    for idx, cls_name in enumerate(imagenet_classes):
        cls_lower = cls_name.lower().replace('_', ' ')
        
        found = -1
        # Check synonyms/heuristics
        for target_cls, keywords in cifar10_synonyms.items():
            for kw in keywords:
                if kw in cls_lower:
                    found = cifar10_classes.index(target_cls)
                    break
            if found != -1:
                break
        
        # Check direct containment
        if found == -1:
            for i, target in enumerate(cifar10_classes):
                if target in cls_lower:
                    found = i
                    break
                    
        # Special dog logic: ImageNet has tons of dogs that don't say "dog"
        # Standard range for dogs in ImageNet 1k: 151-268
        if found == -1:
            if 151 <= idx <= 268:
                found = cifar10_classes.index('dog')
                
        if found != -1:
            mapping_c10[idx] = found

    print(f"Generated {len(mapping_c10)} mappings for CIFAR10")
    
    # Write CIFAR10 mapping
    with open('configs/mappings/imagenet1k_to_cifar10.json', 'w') as f:
        json.dump(mapping_c10, f, indent=4)
        
    # 2. ImageNet -> CIFAR100 (Coarse or Fine? We usually map to fine labels 0-99)
    # CIFAR100 classes are disjoint from CIFAR10 basically.
    # Logic: matching string.
    
    mapping_c100 = {}
    for idx, cls_name in enumerate(imagenet_classes):
        cls_lower = cls_name.lower().replace('_', ' ')
        
        found = -1
        for i, target in enumerate(cifar100_classes):
            target_clean = target.lower().replace('_', ' ')
            # Exact match or strong subset?
            # 'maple_tree' vs 'maple'
            if target_clean == cls_lower or target_clean in cls_lower.split() or cls_lower in target_clean.split():
                 found = i
                 break
                 
        if found != -1:
            mapping_c100[idx] = found
            
    print(f"Generated {len(mapping_c100)} mappings for CIFAR100")
    
    with open('configs/mappings/imagenet1k_to_cifar100.json', 'w') as f:
        json.dump(mapping_c100, f, indent=4)

if __name__ == "__main__":
    generate_mappings()
