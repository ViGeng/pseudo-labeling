import torch
import torch.nn as nn
import torchvision.models as models

from .adapters import ClassificationAdapter

# Optional imports for additional model sources
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


def load_model(name: str, weights_enum: str = "DEFAULT", source: str = "torchvision",
               num_classes: int = None) -> ClassificationAdapter:
    """
    Factory function to load model based on name and weights.
    
    Args:
        name: Model name (e.g., 'resnet18', 'vit_b_16', 'clip_vit_l_14')
        weights_enum: Weight specification ('DEFAULT' for pretrained, or specific weight name)
        source: Model source ('torchvision', 'timm', 'open_clip')
        num_classes: Optional number of output classes (for zero-shot models like CLIP)
    
    Returns:
        ClassificationAdapter wrapping the loaded model
    """
    
    # =========================================================================
    # Torchvision Models
    # =========================================================================
    if source == "torchvision" or source is None:
        model = _load_torchvision_model(name, weights_enum)
        if model is not None:
            return ClassificationAdapter(model)
    
    # =========================================================================
    # Timm Models (BiT, etc.)
    # =========================================================================
    if source == "timm":
        if not TIMM_AVAILABLE:
            raise ImportError("timm package required. Install with: pip install timm")
        model = _load_timm_model(name, weights_enum)
        return ClassificationAdapter(model)
    
    # =========================================================================
    # Open-CLIP Models
    # =========================================================================
    if source == "open_clip":
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("open_clip package required. Install with: pip install open_clip_torch")
        return _load_clip_model(name, weights_enum, num_classes)
    
    raise ValueError(f"Unknown model: {name} from source: {source}")


def _load_torchvision_model(name: str, weights_enum: str) -> nn.Module:
    """Load model from torchvision.models."""
    
    # ResNet family
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if weights_enum == "DEFAULT" else None
        return models.resnet18(weights=weights)
    
    elif name == "resnet101":
        # Cloud model for Resolution & Crop experiments
        # Parameters: ~44.5M | Bias: texture | Role: cloud
        weights = models.ResNet101_Weights.DEFAULT if weights_enum == "DEFAULT" else None
        return models.resnet101(weights=weights)
    
    # MobileNet family
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if weights_enum == "DEFAULT" else None
        return models.mobilenet_v3_large(weights=weights)
    
    # Vision Transformer
    elif name == "vit_b_16":
        # Cloud model for Texture vs Shape experiments
        # Parameters: ~86M | Bias: shape | Role: cloud
        weights = models.ViT_B_16_Weights.DEFAULT if weights_enum == "DEFAULT" else None
        return models.vit_b_16(weights=weights)
    
    # EfficientNet family
    elif name == "efficientnet_b0":
        # Edge model for Specialist vs Generalist experiments
        # Parameters: ~5.3M | Bias: texture | Role: edge
        weights = models.EfficientNet_B0_Weights.DEFAULT if weights_enum == "DEFAULT" else None
        return models.efficientnet_b0(weights=weights)
    
    # ShuffleNet family
    elif name == "shufflenet_v2":
        # Edge model for Robustness Trap experiments (good for AugMix training)
        # Parameters: ~2.3M | Bias: texture | Role: edge
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if weights_enum == "DEFAULT" else None
        return models.shufflenet_v2_x1_0(weights=weights)
    
    # DenseNet family
    elif name == "densenet121":
        # Cloud model for Robustness Trap experiments (standard training, corruption-vulnerable)
        # Parameters: ~8M | Bias: texture | Role: cloud
        weights = models.DenseNet121_Weights.DEFAULT if weights_enum == "DEFAULT" else None
        return models.densenet121(weights=weights)
    
    # Swin Transformer family
    elif name == "swin_t":
        # Hybrid local+global attention model (shifted window transformer)
        # Parameters: ~28M | Bias: hybrid | Role: cloud/edge
        weights = models.Swin_T_Weights.DEFAULT if weights_enum == "DEFAULT" else None
        return models.swin_t(weights=weights)
    
    return None


def _load_timm_model(name: str, weights_enum: str) -> nn.Module:
    """Load model from timm library."""
    
    if name == "bit_resnet152x4":
        # Cloud model for Specialist vs Generalist experiments
        # Parameters: ~936M | Bias: general | Source: ImageNet-21K
        model_name = "resnetv2_152x4_bit.goog_in21k"
        pretrained = weights_enum in ["DEFAULT", "goog_in21k"]
        return timm.create_model(model_name, pretrained=pretrained)
    
    # Generic timm model loading
    pretrained = weights_enum == "DEFAULT"
    return timm.create_model(name, pretrained=pretrained)


def _load_clip_model(name: str, weights_enum: str, num_classes: int = None):
    """
    Load CLIP model from open_clip.
    
    CLIP uses a special adapter for zero-shot classification via text prompts.
    For direct classification, we return a CLIPClassificationAdapter.
    """
    from .adapters import CLIPClassificationAdapter
    
    if name == "clip_vit_l_14":
        # Cloud model for Specialist vs Generalist experiments
        # Parameters: ~428M (vision encoder) | Bias: general | Source: 400M web pairs
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai'
        )
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
        return CLIPClassificationAdapter(model, tokenizer, num_classes)
    
    raise ValueError(f"Unknown CLIP model: {name}")

