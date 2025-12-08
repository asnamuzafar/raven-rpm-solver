"""
Relation Network for Sort-of-CLEVR
===================================
Implementation of "A simple neural network module for relational reasoning"
(Santoro et al., 2017)

Architecture:
1. CNN Encoder: Image -> Feature map -> Object representations
2. Relation Network: Process all object pairs -> Aggregate -> MLP
3. Question conditioning: Concatenate question with each object pair

Expected Results:
- Non-relational: 95%+
- Relational: 95%+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    CNN encoder that extracts spatial feature maps.
    Each spatial location becomes an "object" for the relation network.
    
    Input: (B, 3, 128, 128)
    Output: (B, 24, 5, 5) -> (B, 25, 24+2) with coordinates
    """
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            
            # 64 -> 32
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            
            # 32 -> 16
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            
            # 16 -> 5 (approximately)
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, 128, 128)
        Returns:
            objects: (B, num_objects, features) where objects = spatial locations
        """
        x = self.conv(x)  # (B, 24, H, W)
        
        B, C, H, W = x.shape
        
        # Add coordinate channels
        # Create coordinate grid
        coords_h = torch.linspace(-1, 1, H, device=x.device)
        coords_w = torch.linspace(-1, 1, W, device=x.device)
        coord_h, coord_w = torch.meshgrid(coords_h, coords_w, indexing='ij')
        
        # Expand to batch
        coord_h = coord_h.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        coord_w = coord_w.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        
        # Concatenate
        x = torch.cat([x, coord_h, coord_w], dim=1)  # (B, 26, H, W)
        
        # Flatten spatial dimensions
        x = x.view(B, 26, -1).permute(0, 2, 1)  # (B, H*W, 26)
        
        return x


class RelationNetwork(nn.Module):
    """
    Relation Network (RN) module.
    
    Processes all pairs of objects: RN = sum(g(o_i, o_j, q))
    Then passes through f: output = f(RN)
    """
    def __init__(self, object_dim=26, question_dim=8, hidden_dim=256, output_dim=2):
        super().__init__()
        
        # g_theta: processes object pairs with question
        # Input: (o_i, o_j, q) = 26 + 26 + 8 = 60
        self.g_theta = nn.Sequential(
            nn.Linear(object_dim * 2 + question_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # f_phi: processes aggregated relations
        self.f_phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, objects, question):
        """
        Args:
            objects: (B, N, object_dim) - N object representations
            question: (B, question_dim) - question encoding
        Returns:
            output: (B, output_dim) - answer logits
        """
        B, N, D = objects.shape
        
        # Create all pairs
        # Expand objects for pairing
        obj_i = objects.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        obj_j = objects.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
        
        # Expand question for each pair
        q = question.unsqueeze(1).unsqueeze(1).expand(-1, N, N, -1)  # (B, N, N, Q)
        
        # Concatenate pairs with question
        pairs = torch.cat([obj_i, obj_j, q], dim=-1)  # (B, N, N, 2*D + Q)
        
        # Flatten pairs
        pairs = pairs.view(B, N * N, -1)  # (B, N*N, 2*D + Q)
        
        # Process through g_theta
        relations = self.g_theta(pairs)  # (B, N*N, hidden)
        
        # Sum over all pairs
        relations = relations.sum(dim=1)  # (B, hidden)
        
        # Process through f_phi
        output = self.f_phi(relations)  # (B, output_dim)
        
        return output


class SortOfCLEVRModel(nn.Module):
    """
    Complete model for Sort-of-CLEVR.
    
    Combines CNN encoder with Relation Network.
    """
    def __init__(self, question_dim=8, num_answers=2, hidden_dim=256):
        super().__init__()
        
        self.encoder = CNNEncoder()
        self.rn = RelationNetwork(
            object_dim=26,  # 24 CNN features + 2 coordinates
            question_dim=question_dim,
            hidden_dim=hidden_dim,
            output_dim=num_answers
        )
        
    def forward(self, image, question):
        """
        Args:
            image: (B, 3, 128, 128)
            question: (B, question_dim)
        Returns:
            logits: (B, num_answers)
        """
        objects = self.encoder(image)  # (B, N, 26)
        logits = self.rn(objects, question)  # (B, num_answers)
        return logits


class BaselineCNN(nn.Module):
    """
    Simple CNN+MLP baseline for comparison.
    Does not have relational reasoning capability.
    """
    def __init__(self, question_dim=8, num_answers=2):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8 + question_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_answers),
        )
        
    def forward(self, image, question):
        x = self.conv(image)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, question], dim=1)
        return self.fc(x)


if __name__ == '__main__':
    # Test model
    print("Testing SortOfCLEVRModel...")
    model = SortOfCLEVRModel()
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    
    # Test forward pass
    img = torch.randn(2, 3, 128, 128)
    question = torch.randn(2, 8)
    
    logits = model(img, question)
    print(f"Input: image {img.shape}, question {question.shape}")
    print(f"Output: {logits.shape}")
    print("✓ Model works!")
