"""Curriculum Learning implementations for gradual difficulty increase."""
import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
from typing import Iterator, List, Optional, Callable
import random


class CurriculumSampler(Sampler):
    """
    Curriculum sampler that presents easier samples first.
    Gradually increases difficulty as training progresses.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        num_epochs: int,
        difficulty_fn: Callable = None,
        start_ratio: float = 0.3,
        end_epoch_ratio: float = 0.7,
        seed: int = 42
    ):
        """
        Args:
            dataset: The training dataset
            num_epochs: Total number of training epochs
            difficulty_fn: Function to compute difficulty for each sample
                          If None, uses default based on label length and synthetic flag
            start_ratio: Initial ratio of easiest samples to include (0-1)
            end_epoch_ratio: By this fraction of epochs, all samples are included
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.start_ratio = start_ratio
        self.end_epoch_ratio = end_epoch_ratio
        self.seed = seed
        self.current_epoch = 0
        
        # Compute difficulties
        self.difficulties = self._compute_difficulties(difficulty_fn)
        
        # Sort indices by difficulty
        self.sorted_indices = np.argsort(self.difficulties)
    
    def _compute_difficulties(self, difficulty_fn: Callable = None) -> np.ndarray:
        """Compute difficulty score for each sample."""
        if difficulty_fn is not None:
            return np.array([difficulty_fn(s) for s in self.dataset.samples])
        
        # Default difficulty: based on label length and synthetic flag
        difficulties = []
        for sample in self.dataset.samples:
            label_len = len(sample.get('label', ''))
            is_synthetic = 1.0 if sample.get('is_synthetic', False) else 0.0
            
            # Longer labels = harder, synthetic = harder
            difficulty = label_len / 10.0 + is_synthetic * 0.5
            difficulties.append(difficulty)
        
        return np.array(difficulties)
    
    def set_epoch(self, epoch: int):
        """Update current epoch for curriculum progression."""
        self.current_epoch = epoch
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over sample indices based on curriculum."""
        # Compute progress (0 to 1)
        progress = min(self.current_epoch / (self.num_epochs * self.end_epoch_ratio), 1.0)
        
        # Compute current ratio of samples to use
        current_ratio = self.start_ratio + progress * (1.0 - self.start_ratio)
        
        # Number of samples to include
        n_samples = max(1, int(len(self.dataset) * current_ratio))
        
        # Get easiest n_samples
        selected_indices = self.sorted_indices[:n_samples].tolist()
        
        # Shuffle selected indices
        rng = random.Random(self.seed + self.current_epoch)
        rng.shuffle(selected_indices)
        
        return iter(selected_indices)
    
    def __len__(self) -> int:
        """Return current number of samples (changes with epoch)."""
        progress = min(self.current_epoch / (self.num_epochs * self.end_epoch_ratio), 1.0)
        current_ratio = self.start_ratio + progress * (1.0 - self.start_ratio)
        return max(1, int(len(self.dataset) * current_ratio))


class SelfPacedSampler(Sampler):
    """
    Self-paced learning sampler.
    Samples are weighted by their training loss - harder samples (higher loss)
    are less likely to be selected until the model improves.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        lambda_pace: float = 1.0,
        gamma: float = 1.1,
        min_weight: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            dataset: Training dataset
            lambda_pace: Pacing parameter (higher = include harder samples earlier)
            gamma: Growth factor for lambda_pace per epoch
            min_weight: Minimum sampling weight for any sample
            seed: Random seed
        """
        self.dataset = dataset
        self.lambda_pace = lambda_pace
        self.gamma = gamma
        self.min_weight = min_weight
        self.seed = seed
        self.current_epoch = 0
        
        # Track losses for each sample (updated during training)
        self.sample_losses = np.ones(len(dataset))
    
    def update_losses(self, indices: List[int], losses: List[float]):
        """Update recorded losses for samples."""
        for idx, loss in zip(indices, losses):
            # Exponential moving average
            self.sample_losses[idx] = 0.9 * self.sample_losses[idx] + 0.1 * loss
    
    def set_epoch(self, epoch: int):
        """Update epoch and pace parameter."""
        self.current_epoch = epoch
        # Increase lambda to include harder samples over time
        self.lambda_pace = self.lambda_pace * (self.gamma ** epoch)
    
    def _compute_weights(self) -> np.ndarray:
        """Compute sampling weights based on losses."""
        # Self-paced weight: v = 1 if loss < lambda, else 0
        # Soft version: v = max(0, 1 - loss/lambda)
        weights = np.maximum(self.min_weight, 1.0 - self.sample_losses / self.lambda_pace)
        return weights / weights.sum()
    
    def __iter__(self) -> Iterator[int]:
        """Sample indices weighted by self-paced weights."""
        rng = np.random.RandomState(self.seed + self.current_epoch)
        weights = self._compute_weights()
        
        # Weighted sampling without replacement
        indices = rng.choice(
            len(self.dataset),
            size=len(self.dataset),
            replace=False,
            p=weights
        )
        
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        return len(self.dataset)


class AntiCurriculumSampler(Sampler):
    """
    Anti-curriculum sampler - presents harder samples first.
    Can be useful when model needs to learn difficult patterns early.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        num_epochs: int,
        difficulty_fn: Callable = None,
        hard_ratio: float = 0.3,
        seed: int = 42
    ):
        """
        Args:
            dataset: Training dataset
            num_epochs: Total epochs
            difficulty_fn: Function to compute difficulty
            hard_ratio: Ratio of hardest samples to prioritize
            seed: Random seed
        """
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.hard_ratio = hard_ratio
        self.seed = seed
        self.current_epoch = 0
        
        # Compute difficulties
        if difficulty_fn:
            self.difficulties = np.array([difficulty_fn(s) for s in dataset.samples])
        else:
            self.difficulties = self._default_difficulty()
        
        # Sort by difficulty (descending - hardest first)
        self.sorted_indices = np.argsort(-self.difficulties)
    
    def _default_difficulty(self) -> np.ndarray:
        """Default difficulty computation."""
        difficulties = []
        for sample in self.dataset.samples:
            label_len = len(sample.get('label', ''))
            is_synthetic = 1.0 if sample.get('is_synthetic', False) else 0.0
            difficulty = label_len / 10.0 + is_synthetic * 0.5
            difficulties.append(difficulty)
        return np.array(difficulties)
    
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
    
    def __iter__(self) -> Iterator[int]:
        """Iterate with hard samples appearing more frequently early on."""
        rng = random.Random(self.seed + self.current_epoch)
        
        n_hard = int(len(self.dataset) * self.hard_ratio)
        hard_indices = self.sorted_indices[:n_hard].tolist()
        easy_indices = self.sorted_indices[n_hard:].tolist()
        
        # Weight hard samples more in early epochs
        progress = self.current_epoch / self.num_epochs
        hard_oversample = max(1, int((1 - progress) * 2))  # 2x at start, 1x at end
        
        all_indices = hard_indices * hard_oversample + easy_indices
        rng.shuffle(all_indices)
        
        # Limit to dataset size
        all_indices = all_indices[:len(self.dataset)]
        
        return iter(all_indices)
    
    def __len__(self) -> int:
        return len(self.dataset)


class MixedDifficultySampler(Sampler):
    """
    Sampler that ensures each batch has a mix of easy and hard samples.
    Helps maintain gradient diversity throughout training.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        easy_ratio: float = 0.5,
        difficulty_fn: Callable = None,
        seed: int = 42
    ):
        """
        Args:
            dataset: Training dataset
            batch_size: Batch size
            easy_ratio: Ratio of easy samples per batch
            difficulty_fn: Difficulty computation function
            seed: Random seed
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.easy_ratio = easy_ratio
        self.seed = seed
        self.current_epoch = 0
        
        # Compute difficulties
        if difficulty_fn:
            difficulties = np.array([difficulty_fn(s) for s in dataset.samples])
        else:
            difficulties = self._default_difficulty()
        
        # Split into easy and hard by median
        median_diff = np.median(difficulties)
        self.easy_indices = np.where(difficulties <= median_diff)[0].tolist()
        self.hard_indices = np.where(difficulties > median_diff)[0].tolist()
    
    def _default_difficulty(self) -> np.ndarray:
        difficulties = []
        for sample in self.dataset.samples:
            label_len = len(sample.get('label', ''))
            is_synthetic = 1.0 if sample.get('is_synthetic', False) else 0.0
            difficulty = label_len / 10.0 + is_synthetic * 0.5
            difficulties.append(difficulty)
        return np.array(difficulties)
    
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices ensuring mixed difficulty batches."""
        rng = random.Random(self.seed + self.current_epoch)
        
        easy = self.easy_indices.copy()
        hard = self.hard_indices.copy()
        rng.shuffle(easy)
        rng.shuffle(hard)
        
        # Interleave easy and hard
        n_easy_per_batch = int(self.batch_size * self.easy_ratio)
        n_hard_per_batch = self.batch_size - n_easy_per_batch
        
        indices = []
        easy_idx, hard_idx = 0, 0
        
        while easy_idx < len(easy) or hard_idx < len(hard):
            batch = []
            
            # Add easy samples
            for _ in range(n_easy_per_batch):
                if easy_idx < len(easy):
                    batch.append(easy[easy_idx])
                    easy_idx += 1
            
            # Add hard samples
            for _ in range(n_hard_per_batch):
                if hard_idx < len(hard):
                    batch.append(hard[hard_idx])
                    hard_idx += 1
            
            rng.shuffle(batch)  # Shuffle within batch
            indices.extend(batch)
        
        return iter(indices[:len(self.dataset)])
    
    def __len__(self) -> int:
        return len(self.dataset)


def get_curriculum_sampler(
    sampler_type: str,
    dataset: Dataset,
    **kwargs
) -> Sampler:
    """Factory function to create curriculum sampler.
    
    Args:
        sampler_type: One of 'curriculum', 'self_paced', 'anti', 'mixed'
        dataset: Training dataset
        **kwargs: Additional arguments for the sampler
    
    Returns:
        Sampler instance
    """
    samplers = {
        'curriculum': CurriculumSampler,
        'self_paced': SelfPacedSampler,
        'anti': AntiCurriculumSampler,
        'mixed': MixedDifficultySampler,
    }
    
    if sampler_type not in samplers:
        raise ValueError(f"Unknown sampler type: {sampler_type}. Choose from {list(samplers.keys())}")
    
    return samplers[sampler_type](dataset, **kwargs)
