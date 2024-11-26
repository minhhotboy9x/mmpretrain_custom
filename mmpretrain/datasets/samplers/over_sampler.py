from typing import Iterator
import math

import torch
from mmengine.dataset import DefaultSampler
from typing import Iterator, Optional, Sized

from mmpretrain.registry import DATA_SAMPLERS

@DATA_SAMPLERS\
    .register_module()
class MinorClassOversampler(DefaultSampler):
    """A custom sampler that oversamples minority classes."""

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True,
                 oversample_ratio: float = 2.0) -> None:
        super().__init__(dataset, shuffle, seed, round_up)
        self.oversample_ratio = oversample_ratio
        
        # Compute class frequencies or use some other way to determine minority classes
        self.class_frequencies = self._get_class_frequencies()
        self.minority_classes = self._get_minority_classes()

        # Get the oversample indice of the dataset
        oversampled_indices = []
        for idx in range(len(self.dataset)):
            label = self.dataset.get_cat_ids(idx)[0]
            if label in self.minority_classes:
                oversampled_indices.extend([idx] * int(self.oversample_ratio))
            else:
                oversampled_indices.append(idx)

        self.oversampled_indices = oversampled_indices

        if self.round_up:
            self.num_samples = math.ceil(len(oversampled_indices) / self.world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(oversampled_indices) - self.rank) / self.world_size)
            self.total_size = len(oversampled_indices)
        


    def _get_class_frequencies(self):
        """Compute class frequencies based on dataset annotations."""
        # For simplicity, let's assume the dataset has a method to get labels
        labels = self.dataset.get_gt_labels()
        class_freq = {}
        for label in labels:
            class_freq[label] = class_freq.get(label, 0) + 1
        return class_freq

    def _get_minority_classes(self):
        """Identify minority classes based on frequency."""
        # Here, we simply assume that minority classes have fewer than the mean frequency
        mean_frequency = sum(self.class_frequencies.values()) / len(self.class_frequencies)
        minority_classes = [cls for cls, freq in self.class_frequencies.items() if freq < mean_frequency]
        return minority_classes

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices and oversample minority classes."""
        oversampled_indices = self.oversampled_indices.copy()

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)  # Đảm bảo tính xác định
            shuffled_indices = torch.randperm(len(oversampled_indices), generator=generator) # xáo idx trong oversampled_indices
            oversampled_indices = [oversampled_indices[idx] for idx in shuffled_indices]

        if self.round_up:
            oversampled_indices = (
                oversampled_indices *
                int(self.total_size / len(oversampled_indices) + 1))[:self.total_size]
        
        oversampled_indices = oversampled_indices[self.rank:self.total_size:self.world_size]
        return iter(oversampled_indices)

    def __len__(self) -> int:
        """The number of samples after oversampling."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch