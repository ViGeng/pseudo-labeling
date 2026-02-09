import json
import os
import numpy as np
from typing import Dict, Union, Optional

class LabelMapper:
    """
    Handles mapping from source model labels to target dataset labels.
    """
    def __init__(self, mapping_path: str):
        """
        Args:
            mapping_path (str): Path to the JSON mapping file.
        """
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
            
        with open(mapping_path, 'r') as f:
            raw_map = json.load(f)
            
        # Convert keys and values to integers for efficient lookup
        self.mapping = {int(k): int(v) for k, v in raw_map.items()}
        
        # Pre-compute max index for array-based lookup if feasible, 
        # but dictionary lookup is safer for sparse or large indices. 
        # given 1000 classes, array lookup is fast.
        self.max_key = max(self.mapping.keys())
        self.lookup_array = np.full(self.max_key + 1, -1, dtype=int)
        for k, v in self.mapping.items():
            self.lookup_array[k] = v

    def map(self, predictions: np.ndarray) -> np.ndarray:
        """
        Maps model predictions to dataset labels.
        Unknown classes are mapped to -1.
        
        Args:
            predictions (np.ndarray): 1D array of predicted class indices.
            
        Returns:
            np.ndarray: Mapped class indices.
        """
        # Efficient mapping using numpy indexing for valid range
        # 1. Mask out values larger than our lookup table
        valid_mask = (predictions >= 0) & (predictions <= self.max_key)
        
        # 2. Initialize result with -1
        result = np.full_like(predictions, -1)
        
        # 3. Map valid entries
        # valid predictions are used as indices into lookup_array
        if np.any(valid_mask):
            result[valid_mask] = self.lookup_array[predictions[valid_mask]]
            
        return result
