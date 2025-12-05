"""
Dataset utilities for RAVEN puzzles

Handles loading and preprocessing of RAVEN .npz files.
"""
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
from scipy import ndimage


class RAVENAugmentation:
    """
    Data augmentation for RAVEN puzzles.
    Applies consistent augmentation across all 16 panels while preserving
    the relational structure (only augments in ways that don't change the answer).
    """
    def __init__(self, 
                 noise_prob: float = 0.5,
                 noise_std: float = 0.1,
                 brightness_prob: float = 0.5,
                 brightness_range: Tuple[float, float] = (0.7, 1.3),
                 contrast_prob: float = 0.5,
                 contrast_range: Tuple[float, float] = (0.7, 1.3)):
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.brightness_prob = brightness_prob
        self.brightness_range = brightness_range
        self.contrast_prob = contrast_prob
        self.contrast_range = contrast_range
        
    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to all panels consistently.
        
        Args:
            imgs: (16, H, W) float array normalized to [0, 1]
        Returns:
            Augmented images with same shape
        """
        # Gaussian noise (same for all panels to preserve relations)
        if np.random.random() < self.noise_prob:
            noise = np.random.normal(0, self.noise_std, imgs.shape).astype(np.float32)
            imgs = imgs + noise
        
        # Brightness adjustment (same factor for all panels)
        if np.random.random() < self.brightness_prob:
            factor = np.random.uniform(*self.brightness_range)
            imgs = imgs * factor
        
        # Contrast adjustment (same for all panels)
        if np.random.random() < self.contrast_prob:
            factor = np.random.uniform(*self.contrast_range)
            mean = imgs.mean()
            imgs = (imgs - mean) * factor + mean
        
        # Clamp to valid range
        imgs = np.clip(imgs, 0, 1)
        
        return imgs


class RAVENDataset(Dataset):
    """
    PyTorch Dataset for RAVEN puzzles.
    
    Each puzzle contains:
    - 16 images: 8 context panels + 8 answer choices
    - 1 target: index of correct answer (0-7)
    """
    def __init__(self, files: List[Path], transform=None, augment: bool = False, return_meta: bool = False):
        """
        Args:
            files: List of paths to .npz files
            transform: Optional transform to apply to images
            augment: Whether to apply data augmentation (for training)
            return_meta: Whether to return ground-truth metadata for supervised learning
        """
        self.files = [str(f) for f in files]
        self.transform = transform
        self.augmentation = RAVENAugmentation() if augment else None
        self.return_meta = return_meta
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int):
        """
        Returns:
            images: (16, 160, 160) float tensor normalized to [0, 1]
            target: scalar tensor with correct answer index
            path: path to the source file
            meta (optional): dict with ground-truth attributes if return_meta=True
        """
        path = self.files[idx]
        data = np.load(path)
        
        # Load images and normalize to [0, 1]
        imgs = data["image"].astype(np.float32) / 255.0  # (16, 160, 160)
        
        # Apply augmentation if enabled (training only)
        if self.augmentation is not None:
            imgs = self.augmentation(imgs)
        
        if self.transform:
            imgs = self.transform(imgs)
        
        x = torch.from_numpy(imgs)
        y = torch.tensor(int(data["target"]), dtype=torch.long)
        
        if self.return_meta:
            # Load ground-truth metadata for supervised attribute extraction
            meta = self._extract_meta(data)
            return x, y, path, meta
        
        return x, y, path
    
    def _extract_meta(self, data) -> dict:
        """
        Extract ground-truth attributes from I-RAVEN metadata.
        
        meta_matrix shape: (num_attributes, 9) for 9 panels in 3x3 grid
        meta_target shape: (num_attributes,) for the correct answer panel
        
        I-RAVEN attributes (from Attribute.py):
        - For single-object configs: Type, Size, Color (indices 0-2)
        - For multi-object configs: Number, Position, Type, Size, Color
        """
        meta = {}
        
        if 'meta_matrix' in data.keys() and 'meta_target' in data.keys():
            meta_matrix = data['meta_matrix']  # (num_attrs, 9)
            meta_target = data['meta_target']  # (num_attrs,)
            
            # Build full 16-panel attribute matrix
            # Context panels (0-7) correspond to positions 0-7 in meta_matrix
            # But meta_matrix has 9 entries (3x3 grid), position 8 is the answer
            
            # Context panels: first 8 context positions (0-7)
            # Note: In 3x3 grid, position 8 is bottom-right (the answer position)
            # We need to map context panels to meta_matrix indices
            
            num_attrs = meta_matrix.shape[0]
            
            # Context: panels 0-7 (positions in 3x3 minus bottom-right)
            # The 8 context panels map to: row0=[0,1,2], row1=[3,4,5], row2_partial=[6,7]
            context_attrs = meta_matrix[:, :8]  # (num_attrs, 8)
            
            # For each choice, create the full attribute set
            # Choice i has meta_target as the answer panel attributes
            # But we also have access to predict array which has all 8 candidate predictions
            
            # Store context attributes
            meta['context'] = torch.from_numpy(context_attrs.astype(np.int64))  # (num_attrs, 8)
            
            # The correct answer's attributes
            meta['target_attrs'] = torch.from_numpy(meta_target.astype(np.int64))  # (num_attrs,)
            
            # Store raw meta_matrix for rule detection  
            meta['meta_matrix'] = torch.from_numpy(meta_matrix.astype(np.int64))  # (num_attrs, 9)
            
        if 'structure' in data.keys():
            # Structure encodes the configuration type
            meta['structure'] = data['structure']
            
        return meta


def get_split(path: Path) -> str:
    """Extract split name (train/val/test) from filename"""
    m = re.search(r"_(train|val|test)\.npz$", str(path))
    return m.group(1) if m else "unknown"


def get_split_files(data_dir: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Get train/val/test file lists from a data directory.
    
    Args:
        data_dir: Path to directory containing .npz files
        
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.rglob("*.npz"))
    
    if not npz_files:
        raise ValueError(f"No .npz files found in {data_dir}")
    
    train_files = [p for p in npz_files if get_split(p) == "train"]
    val_files = [p for p in npz_files if get_split(p) == "val"]
    test_files = [p for p in npz_files if get_split(p) == "test"]
    
    # Fallback: split by ratio if no split tags found
    if not train_files:
        n = len(npz_files)
        train_files = npz_files[:int(0.8 * n)]
        val_files = npz_files[int(0.8 * n):int(0.9 * n)]
        test_files = npz_files[int(0.9 * n):]
    
    return train_files, val_files, test_files


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 16,
    num_workers: int = 2,
    pin_memory: bool = True,
    return_meta: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory (useful for GPU)
        return_meta: Whether to return metadata for supervised attribute prediction
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_files, val_files, test_files = get_split_files(data_dir)
    
    # Enable augmentation for training set only
    train_ds = RAVENDataset(train_files, augment=True, return_meta=return_meta)
    val_ds = RAVENDataset(val_files, augment=False, return_meta=return_meta)
    test_ds = RAVENDataset(test_files, augment=False, return_meta=return_meta)
    
    def collate_fn(batch):
        """Custom collate to handle metadata dict."""
        if len(batch[0]) == 4:  # x, y, path, meta
            x_list, y_list, path_list, meta_list = zip(*batch)
            
            x = torch.stack(x_list)
            y = torch.stack(y_list)
            
            # Collate metadata
            meta = {}
            if meta_list[0]:
                for key in meta_list[0].keys():
                    if key == 'structure':
                        meta[key] = [m[key] for m in meta_list]
                    elif isinstance(meta_list[0][key], torch.Tensor):
                        meta[key] = torch.stack([m[key] for m in meta_list])
            
            return x, y, path_list, meta
        else:
            x_list, y_list, path_list = zip(*batch)
            return torch.stack(x_list), torch.stack(y_list), path_list
            
    # Use custom collate if returning metadata, otherwise default
    collate_func = collate_fn if return_meta else None
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_func
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_func
    )
    test_dl = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_func
    )
    
    print(f"Dataset splits: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    return train_dl, val_dl, test_dl


def load_single_puzzle(path: str) -> Tuple[np.ndarray, int]:
    """
    Load a single puzzle for visualization/inference.
    
    Args:
        path: Path to .npz file
        
    Returns:
        images: (16, 160, 160) uint8 array
        target: correct answer index
    """
    data = np.load(path)
    return data["image"], int(data["target"])

