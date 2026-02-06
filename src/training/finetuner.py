import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

class FineTuner:
    """
    Handles fine-tuning of pretrained models for new target datasets.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dictionary containing fine-tuning parameters:
                - epochs: int
                - lr: float
                - weight_decay: float
                - momentum: float (for SGD)
                - freeze_backbone: bool
                - device: str
        """
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    def tune(self, 
             model: nn.Module, 
             train_loader: DataLoader, 
             val_loader: Optional[DataLoader] = None,
             source_dataset: str = "imagenet",
             target_dataset: str = "cifar10",
             num_classes: int = 10) -> nn.Module:
        """
        Fine-tunes the model if source and target datasets differ.
        
        Args:
            model: The pretrained model.
            train_loader: DataLoader for the target training set.
            val_loader: Optional validation loader.
            source_dataset: Name of the dataset the model was pretrained on.
            target_dataset: Name of the target dataset.
            num_classes: Number of classes in the target dataset.
            
        Returns:
            The fine-tuned model.
        """
        if source_dataset.lower() == target_dataset.lower():
            logger.info(f"Source and target datasets are the same ({source_dataset}). Skipping fine-tuning.")
            return model

        logger.info(f"Starting fine-tuning: {source_dataset} -> {target_dataset}")
        
        model = model.to(self.device)
        
        # 1. Replace Head
        self._replace_head(model, num_classes)
        model = model.to(self.device) # Ensure new head is on device

        # 2. Freeze Backbone (Linear Probe) or Full Finetune
        if self.config.get("freeze_backbone", True):
            self._freeze_backbone(model)
        
        # 3. Train
        self._train_loop(model, train_loader, val_loader)
        
        return model

    def _replace_head(self, model: nn.Module, num_classes: int):
        """
        Smartly replaces the classification head of common architectures.
        """
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            # ResNet case
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            logger.info(f"Replaced ResNet fc layer: {in_features} -> {num_classes}")
            
        elif hasattr(model, 'classifier'):
            # MobileNet / VGG / DenseNet case
            if isinstance(model.classifier, nn.Sequential):
                # MobileNetV3 typically has a sequential classifier. 
                # The last layer is usually the projection / linear layer.
                # We iterate to find the last linear layer.
                for i in range(len(model.classifier) - 1, -1, -1):
                    if isinstance(model.classifier[i], nn.Linear):
                        in_features = model.classifier[i].in_features
                        model.classifier[i] = nn.Linear(in_features, num_classes)
                        logger.info(f"Replaced MobileNet/Classifier layer at index {i}: {in_features} -> {num_classes}")
                        break
            elif isinstance(model.classifier, nn.Linear):
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
                logger.info(f"Replaced Classifier linear layer: {in_features} -> {num_classes}")
        
        elif hasattr(model, 'heads') and hasattr(model.heads, 'head'):
             # ViT (torchvision version) sometimes
             if isinstance(model.heads.head, nn.Linear):
                in_features = model.heads.head.in_features
                model.heads.head = nn.Linear(in_features, num_classes)
                logger.info(f"Replaced ViT head: {in_features} -> {num_classes}")

        else:
            logger.warning("Could not automatically detect classification head. Fine-tuning might fail or require updates.")

    def _freeze_backbone(self, model: nn.Module):
        """
        Freezes all parameters except the (newly initialized) head.
        This relies on the fact that we just replaced the head, so its requires_grad is True by default.
        """
        for name, param in model.named_parameters():
             # Heuristic: if 'fc' or 'classifier' or 'head' is in name, keep it trainable.
             # A safer way is checking if the param object is checking the id of the new head parameters, 
             # but string matching is robust enough for standard models.
             if "fc" in name or "classifier" in name or "head" in name:
                 param.requires_grad = True
             else:
                 param.requires_grad = False
        
        logger.info("Backbone frozen. Only head is trainable.")

    def _train_loop(self, model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader]):
        epochs = self.config.get("epochs", 5)
        lr = self.config.get("lr", 0.001)
        momentum = self.config.get("momentum", 0.9)
        
        # Only optimize parameters that require grad
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=self.config.get("weight_decay", 1e-4))
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            epoch_loss = running_loss / total
            epoch_acc = 100. * correct / total
            
            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
            
            if val_loader:
                self._validate(model, val_loader, criterion)
                model.train() # Switch back to train mode

    def _validate(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        val_loss = running_loss / total
        val_acc = 100. * correct / total
        logger.info(f"Validation | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
