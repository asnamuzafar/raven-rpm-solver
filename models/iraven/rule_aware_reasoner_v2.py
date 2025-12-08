"""
Rule-Aware Reasoner V2 (Regularized)
=====================================
Simplified architecture to prevent overfitting.
1. SimpleSpatialEncoder (4-layer CNN)
2. Global Average Pooling (Aggressive abstraction)
3. MLP Rule Predictor (Dropout=0.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder_simple import SimpleSpatialEncoder
# We need SupervisedAttributeHead if we want attribute loss
from models.rule_reasoner import SupervisedAttributeHead

class RuleAwareReasonerV2(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # 1. Image Encoder
        # Input: (B, 1, 160, 160)
        # Output: (B, 512, 5, 5)
        self.encoder = SimpleSpatialEncoder(output_dim=512)
        
        # 2. Pooling to force abstraction (lose precise pixels)
        self.pool = nn.AdaptiveAvgPool2d(1) # (B, 512, 1, 1)
        
        # 3. Rule Predictor
        # Input: 3 panels * 512 features = 1536
        # Output: 128 (Rule Embedding)
        self.rule_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), # Regularization
            nn.Linear(512, 128)
        )
        
        # 4. Validity Scorer (Is this rule valid?)
        self.validity_scorer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        
        # 5. Attribute Head (for auxiliary supervision)
        # Input: 512 features
        self.attr_head = SupervisedAttributeHead(512, 128)
        
    def forward(self, context, choices, return_extras=False):
        """
        Args:
            context: (B, 8, 1, 160, 160) - Raw Images! 
            choices: (B, 8, 1, 160, 160) - Raw Images!
            
            Wait, train_v2.py calls:
             logits, extras = model(x, ...)
             where model is Rule-Aware-Reasoner directly? 
             No, train_v2.py has a wrapper `RuleAwareModelV2`? 
             
             Let's check train_v2.py again.
             Lines 243: model = RuleAwareReasonerV2(device=device).to(device)
             
             So train_v2.py passes 'x' (Batch of images).
             x in train_v2 is (B, 16, 1, 160, 160).
        """
        # Unwrap images
        # x is usually (B, 16, 1, 160, 160)
        # We need to distinguish if we receive pre-encoded features or raw images.
        # But 'RuleAwareReasonerV2' now HAS internal encoder.
        # So inputs must be raw images.
        pass

    def forward(self, x, return_extras=False):
        # x: (B, 16, 1, 160, 160) or (B, 16, 160, 160)
        if x.dim() == 4:
            x = x.unsqueeze(2)
            
        B, N, C, H, W = x.shape
        
        # 1. Encode All Panels
        x_flat = x.view(B*N, C, H, W)
        feats_map = self.encoder(x_flat) # (B*16, 512, 5, 5)
        feats = self.pool(feats_map).view(B, N, 512) # (B, 16, 512)
        
        # Split Context (8) and Choices (8)
        ctx = feats[:, :8, :]     # (B, 8, 512)
        choices = feats[:, 8:, :] # (B, 8, 512)
        
        # Helper to get rule embedding
        def get_rule(p1, p2, p3):
            # p: (B, 512)
            inp = torch.cat([p1, p2, p3], dim=1) # (B, 1536)
            return self.rule_net(inp) # (B, 128)
            
        # P0 P1 P2
        # P3 P4 P5
        # P6 P7 ?
        
        p = [ctx[:, i] for i in range(8)]
        
        # Row 0, Row 1
        r0 = get_rule(p[0], p[1], p[2])
        r1 = get_rule(p[3], p[4], p[5])
        
        # Col 0, Col 1
        c0 = get_rule(p[0], p[3], p[6])
        c1 = get_rule(p[1], p[4], p[7])
        
        # Score Choices
        scores = []
        r2_list = []
        c2_list = []
        
        # Precompute context rules for efficiency?
        # r0, r1 are fixed.
        
        for i in range(8):
            choice_feat = choices[:, i] # (B, 512)
            
            # Form Row 2 and Col 2
            r2 = get_rule(p[6], p[7], choice_feat)
            c2 = get_rule(p[2], p[5], choice_feat)
            
            # Validity
            v_score = self.validity_scorer(r2) + self.validity_scorer(c2)
            
            # Consistency (-MSE)
            # r2 should match r1? or r0?
            # Standard: r2 ~ r1.
            c_score = -F.mse_loss(r2, r1, reduction='none').mean(1, keepdim=True) \
                      -F.mse_loss(c2, c1, reduction='none').mean(1, keepdim=True)
                      
            scores.append(v_score + c_score)
            
            if return_extras:
                r2_list.append(r2)
                c2_list.append(c2)
                
        logits = torch.cat(scores, dim=1)
        
        if return_extras:
            extras = {}
            extras['r0'] = r0
            extras['r1'] = r1
            extras['c0'] = c0
            extras['c1'] = c1
            extras['r2_stack'] = torch.stack(r2_list, dim=1)
            extras['c2_stack'] = torch.stack(c2_list, dim=1)
            
            # Attributes
            # Flatten context features: (B*8, 512)
            attr_logits = self.attr_head(ctx.reshape(-1, 512))
            for k, v in attr_logits.items():
                extras[f'attr_{k}'] = v.view(B, 8, -1)
                
            return logits, extras
            
        return logits
