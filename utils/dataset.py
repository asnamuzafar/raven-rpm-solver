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


class RAVENDataset(Dataset):
    """
    PyTorch Dataset for RAVEN puzzles.
    
    Each puzzle contains:
    - 16 images: 8 context panels + 8 answer choices
    - 1 target: index of correct answer (0-7)
    """
    def __init__(self, files: List[Path], transform=None):
        """
        Args:
            files: List of paths to .npz files
            transform: Optional transform to apply to images
        """
        self.files = [str(f) for f in files]
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            images: (16, 160, 160) float tensor normalized to [0, 1]
            target: scalar tensor with correct answer index
            path: path to the source file
        """
        path = self.files[idx]
        data = np.load(path)
        
        # Load images and normalize to [0, 1]
        imgs = data["image"].astype(np.float32) / 255.0  # (16, 160, 160)
        
        if self.transform:
            imgs = self.transform(imgs)
        
        x = torch.from_numpy(imgs)
        y = torch.tensor(int(data["target"]), dtype=torch.long)
        
        return x, y, path


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
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory (useful for GPU)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_files, val_files, test_files = get_split_files(data_dir)
    
    train_ds = RAVENDataset(train_files)
    val_ds = RAVENDataset(val_files)
    test_ds = RAVENDataset(test_files)
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_dl = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
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

