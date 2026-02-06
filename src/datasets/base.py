from abc import ABC, abstractmethod
from typing import Tuple, Any
from torch.utils.data import Dataset

class BaseDataset(Dataset, ABC):
    """
    Abstract Base Class for all datasets in the labeling project.
    Ensures that __getitem__ returns (sample_id, image, target).
    """

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, Any, Any]:
        """
        Args:
            index (int): Index
        
        Returns:
            tuple: (sample_id, image, target) where target is class_index or similar.
                   sample_id must be a string unique to the sample.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
