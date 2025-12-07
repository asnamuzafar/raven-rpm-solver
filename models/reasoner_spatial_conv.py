"""
Spatial Convolutional Reasoner (Phase 5)

Addresses the overfitting observed with the SpatialAdapter (flattening).
Instead of flattening 5x5 spatial maps into absolute position vectors,
this reasoner applies Convolutions over the feature maps directly.

Key Idea:
- Translational Invariance: A rule like "shift right" is a local operation 
  that should be detected regardless of absolute position.
- Stacked Convolutions: 3 panels in a row are stacked depthwise (C*3, 5, 5).
- 2D Conv layers process this stack to output a relation score.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class relational_conv_block(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        # Global Average Pooling to aggregate spatial relation detection
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # MLP to score the detected relation
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*8, 3*C, 5, 5)
        x = self.conv(x)      # (B*8, H, 5, 5)
        x = self.avgpool(x)   # (B*8, H, 1, 1)
        x = x.view(x.size(0), -1) # (B*8, H)
        score = self.scorer(x)    # (B*8, 1)
        return score

class SpatialConvolutionalReasoner(nn.Module):
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        # Input has 3 panels stacked -> 3 * feature_dim channels
        self.row_conv = relational_conv_block(feature_dim * 3, hidden_dim)
        self.col_conv = relational_conv_block(feature_dim * 3, hidden_dim)
        
    def forward(self, context: torch.Tensor, choices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, 8, C, H, W) -> spatial maps
            choices: (B, 8, C, H, W)
        Returns:
            logits: (B, 8)
        """
        B, N_ctx, C, H, W = context.shape
        _, N_choice, _, _, _ = choices.shape
        
        # We need to score each choice.
        # Vectorized approach:
        # P = Panels. p0..p7 are context. p8 is choice c.
        
        p = [context[:, i] for i in range(8)] # List of (B, C, H, W)
        
        # Prepare Rows/Cols that don't depend on choice (Row0, Row1, Col0, Col1)
        # But wait, purely relational networks usually just score the COMPLETION.
        # i.e. Does Row2 form a valid relation? Does Col2 form a valid relation?
        # Standard RN sums up compatibility of all simple relations.
        # Here we only care if the choice completes the pattern.
        # So we evaluate Row2(choice) and Col2(choice).
        # We implicitly assume Row0/Row1 are valid examples of the rule?
        # Actually, strict RNs often compare Row2Score vs Row0Score.
        # But simpler "compatibility" scoring usually sufficient: Score(Row2) + Score(Col2).
        
        # Let's compute scores for Row2 and Col2 for each choice.
        
        scores = []
        for c_idx in range(8):
            choice = choices[:, c_idx] # (B, C, H, W)
            
            # Row 2: p6, p7, choice
            row2 = torch.cat([p[6], p[7], choice], dim=1) # (B, 3C, H, W)
            
            # Col 2: p2, p5, choice
            col2 = torch.cat([p[2], p[5], choice], dim=1) # (B, 3C, H, W)
            
            # Compute scores
            r_score = self.row_conv(row2) # (B, 1)
            c_score = self.col_conv(col2) # (B, 1)
            
            scores.append(r_score + c_score)
            
        logits = torch.cat(scores, dim=1) # (B, 8)
        return logits
