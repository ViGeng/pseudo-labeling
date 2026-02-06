import torch
import torchvision.models as models
from .adapters import ClassificationAdapter

def load_model(name: str, weights_enum: str = "DEFAULT") -> ClassificationAdapter:
    """
    Factory function to load model based on name and weights.
    """
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if weights_enum == "DEFAULT" else None
        model = models.resnet18(weights=weights)
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if weights_enum == "DEFAULT" else None
        model = models.mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Unknown model: {name}")
    
    # Move to device is handled in the main runner usually, but we can do it here if needed.
    # For now, just wrap it.
    return ClassificationAdapter(model)
