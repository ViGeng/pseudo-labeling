import torch
import torch.nn as nn
from typing import Dict, Any

class ClassificationAdapter(nn.Module):
    """
    Wraps a classification model to ensure consistent output format.
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
