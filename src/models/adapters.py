from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class ClassificationAdapter(nn.Module):
    """
    Wraps a classification model to ensure consistent output format.
    Works with standard torchvision/timm models that output logits.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
        
        return {
            "logits": logits,
            "predictions": predictions,
            "confidences": confidences
        }


class CLIPClassificationAdapter(nn.Module):
    """
    Adapter for CLIP models to perform zero-shot classification.
    
    CLIP classifies by computing similarity between image embeddings 
    and text embeddings of class names (e.g., "a photo of a dog").
    
    Usage:
        adapter = CLIPClassificationAdapter(clip_model, tokenizer)
        adapter.set_class_names(["dog", "cat", "bird"])  # Call before inference
        outputs = adapter(images)
    """
    
    # Common prompt templates for zero-shot classification
    DEFAULT_TEMPLATES = [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a photo of the {}.",
        "a good photo of a {}.",
        "a rendition of a {}.",
        "a photo of the small {}.",
        "a photo of the large {}.",
    ]
    
    def __init__(self, model: nn.Module, tokenizer, num_classes: Optional[int] = None,
                 templates: List[str] = None):
        super().__init__()
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.num_classes = num_classes
        self.templates = templates or self.DEFAULT_TEMPLATES
        
        # Text embeddings cache (computed once per class set)
        self._text_embeddings = None
        self._class_names = None
    
    def set_class_names(self, class_names: List[str]):
        """
        Set the class names for zero-shot classification.
        Must be called before forward() if class names change.
        """
        self._class_names = class_names
        self._text_embeddings = None  # Force recomputation
        self.num_classes = len(class_names)
    
    def _compute_text_embeddings(self, device):
        """Compute and cache text embeddings for all class names."""
        if self._class_names is None:
            raise ValueError("Class names not set. Call set_class_names() first.")
        
        all_embeddings = []
        with torch.no_grad():
            for class_name in self._class_names:
                # Create prompts from templates
                prompts = [template.format(class_name) for template in self.templates]
                tokens = self.tokenizer(prompts).to(device)
                
                # Get text embeddings and average across templates
                text_features = self.model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.mean(dim=0)
                text_features = text_features / text_features.norm()
                all_embeddings.append(text_features)
        
        self._text_embeddings = torch.stack(all_embeddings, dim=0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = x.device
        
        # Compute text embeddings if needed
        if self._text_embeddings is None:
            self._compute_text_embeddings(device)
        
        text_embeddings = self._text_embeddings.to(device)
        
        with torch.no_grad():
            # Get image embeddings
            image_features = self.model.encode_image(x)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity (logits)
            logits = 100.0 * image_features @ text_embeddings.T
            probs = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
        
        return {
            "logits": logits,
            "predictions": predictions,
            "confidences": confidences
        }

