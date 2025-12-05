"""
Contrastive Losses for RAVEN Neuro-Symbolic Reasoning

Implements SOTA loss functions inspired by:
- CoPINet: Contrastive Perceptual Inference
- DCNet: Dual-Contrast Network
- SRAN: Stratified Rule-Aware Network

These losses help the model:
1. Discriminate between correct and incorrect answers (contrastive)
2. Learn ranking relationships between choices (ranking)
3. Enforce consistency across rows/columns (consistency)
4. Predict rule types as auxiliary task (auxiliary)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss that pulls correct answer embedding closer to context
    while pushing incorrect answers away.
    
    Inspired by CoPINet: "Learning Perceptual Inference by Contrasting"
    
    L = -log(exp(sim(ctx, correct)) / sum(exp(sim(ctx, choice_i))))
    
    This is essentially a temperature-scaled cross-entropy over similarities.
    """
    def __init__(self, temperature: float = 0.1, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(
        self, 
        context_repr: torch.Tensor,  # (B, D) - encoded context pattern
        choice_reprs: torch.Tensor,  # (B, 8, D) - encoded choices
        targets: torch.Tensor        # (B,) - correct answer indices
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            context_repr: Encoded representation of all 8 context panels
            choice_reprs: Encoded representations of each choice
            targets: Ground truth answer indices
        """
        B = context_repr.shape[0]
        
        # Normalize for cosine similarity
        context_norm = F.normalize(context_repr, dim=-1)  # (B, D)
        choices_norm = F.normalize(choice_reprs, dim=-1)  # (B, 8, D)
        
        # Compute similarities between context and all choices
        # context_norm: (B, D) -> (B, 1, D)
        # choices_norm: (B, 8, D)
        # similarities: (B, 8)
        similarities = torch.bmm(
            context_norm.unsqueeze(1),  # (B, 1, D)
            choices_norm.transpose(1, 2)  # (B, D, 8)
        ).squeeze(1)  # (B, 8)
        
        # Scale by temperature
        logits = similarities / self.temperature
        
        # Cross-entropy loss over choices (correct answer should have highest similarity)
        loss = F.cross_entropy(logits, targets)
        
        return loss


class RankingLoss(nn.Module):
    """
    Margin-based ranking loss for answer selection.
    
    Ensures correct answer scores higher than all incorrect answers by a margin.
    
    L = sum over incorrect answers: max(0, margin - (score_correct - score_wrong))
    
    This is a multi-class hinge loss that focuses on hard negatives.
    """
    def __init__(self, margin: float = 1.0, hard_negative_weight: float = 2.0):
        super().__init__()
        self.margin = margin
        self.hard_negative_weight = hard_negative_weight
        
    def forward(
        self, 
        logits: torch.Tensor,  # (B, 8) - scores for each choice
        targets: torch.Tensor  # (B,) - correct answer indices
    ) -> torch.Tensor:
        """
        Compute ranking loss with hard negative mining.
        """
        B = logits.shape[0]
        device = logits.device
        
        # Get correct answer scores
        correct_scores = logits.gather(1, targets.unsqueeze(1))  # (B, 1)
        
        # Create mask for incorrect answers
        incorrect_mask = torch.ones_like(logits, dtype=torch.bool)
        incorrect_mask.scatter_(1, targets.unsqueeze(1), False)
        
        # Compute margin violations for all incorrect answers
        # margin_violation = max(0, margin - (correct - incorrect))
        margin_violations = F.relu(
            self.margin - (correct_scores - logits)
        )  # (B, 8)
        
        # Zero out the correct answer's contribution
        margin_violations = margin_violations * incorrect_mask.float()
        
        # Hard negative mining: weight higher violations more
        # (these are the confusing wrong answers)
        if self.hard_negative_weight > 1.0:
            # Find hardest negative (highest score among wrong answers)
            wrong_scores = logits.masked_fill(~incorrect_mask, float('-inf'))
            hardest_idx = wrong_scores.argmax(dim=1, keepdim=True)
            
            # Create weight mask
            weights = torch.ones_like(margin_violations)
            weights.scatter_(1, hardest_idx, self.hard_negative_weight)
            
            margin_violations = margin_violations * weights
        
        # Average over batch and choices
        loss = margin_violations.sum() / (B * 7)  # 7 incorrect per sample
        
        return loss


class ConsistencyLoss(nn.Module):
    """
    Row/Column/Diagonal consistency loss.
    
    Enforces that the inferred pattern from rows 0,1 should match row 2,
    and similarly for columns and diagonals.
    
    L = ||pattern(row0) - pattern(row1)||^2 + ||pattern(row1) - pattern(row2)||^2
    
    This encourages the model to learn consistent transformations.
    """
    def __init__(self, weight_row: float = 1.0, weight_col: float = 1.0, 
                 weight_diag: float = 0.5):
        super().__init__()
        self.weight_row = weight_row
        self.weight_col = weight_col
        self.weight_diag = weight_diag
        
    def forward(
        self, 
        context_features: torch.Tensor,  # (B, 8, D) - features of context panels
        choice_features: torch.Tensor,   # (B, 8, D) - features of choices
        targets: torch.Tensor            # (B,) - correct answer indices
    ) -> torch.Tensor:
        """
        Compute consistency loss for row/column patterns.
        
        Grid positions: 0,1,2 (row0), 3,4,5 (row1), 6,7,X (row2)
        """
        B, _, D = context_features.shape
        device = context_features.device
        
        # Get correct choice features
        correct_choice = choice_features.gather(
            1, targets.unsqueeze(1).unsqueeze(2).expand(-1, -1, D)
        ).squeeze(1)  # (B, D)
        
        # === Row patterns (difference between consecutive elements) ===
        # Row 0: positions 0,1,2
        row0_diff1 = context_features[:, 1] - context_features[:, 0]  # (B, D)
        row0_diff2 = context_features[:, 2] - context_features[:, 1]
        
        # Row 1: positions 3,4,5
        row1_diff1 = context_features[:, 4] - context_features[:, 3]
        row1_diff2 = context_features[:, 5] - context_features[:, 4]
        
        # Row 2: positions 6,7,correct_choice
        row2_diff1 = context_features[:, 7] - context_features[:, 6]
        row2_diff2 = correct_choice - context_features[:, 7]
        
        # Row consistency: patterns should be similar across rows
        row_loss = (
            F.mse_loss(row0_diff1, row1_diff1) +
            F.mse_loss(row0_diff2, row1_diff2) +
            F.mse_loss(row1_diff1, row2_diff1) +
            F.mse_loss(row1_diff2, row2_diff2)
        ) / 4
        
        # === Column patterns ===
        # Col 0: positions 0,3,6
        col0_diff1 = context_features[:, 3] - context_features[:, 0]
        col0_diff2 = context_features[:, 6] - context_features[:, 3]
        
        # Col 1: positions 1,4,7
        col1_diff1 = context_features[:, 4] - context_features[:, 1]
        col1_diff2 = context_features[:, 7] - context_features[:, 4]
        
        # Col 2: positions 2,5,correct_choice
        col2_diff1 = context_features[:, 5] - context_features[:, 2]
        col2_diff2 = correct_choice - context_features[:, 5]
        
        col_loss = (
            F.mse_loss(col0_diff1, col1_diff1) +
            F.mse_loss(col0_diff2, col1_diff2) +
            F.mse_loss(col1_diff1, col2_diff1) +
            F.mse_loss(col1_diff2, col2_diff2)
        ) / 4
        
        # === Diagonal patterns ===
        # Main diagonal: 0,4,correct_choice
        main_diag_diff1 = context_features[:, 4] - context_features[:, 0]
        main_diag_diff2 = correct_choice - context_features[:, 4]
        
        # Anti-diagonal: 2,4,6
        anti_diag_diff1 = context_features[:, 4] - context_features[:, 2]
        anti_diag_diff2 = context_features[:, 6] - context_features[:, 4]
        
        diag_loss = (
            F.mse_loss(main_diag_diff1, main_diag_diff2) +
            F.mse_loss(anti_diag_diff1, anti_diag_diff2)
        ) / 2
        
        # Weighted combination
        total_loss = (
            self.weight_row * row_loss +
            self.weight_col * col_loss +
            self.weight_diag * diag_loss
        )
        
        return total_loss


class AuxiliaryRuleLoss(nn.Module):
    """
    Auxiliary loss for rule type prediction.
    
    Predicts which rule governs each attribute (shape, size, color, count):
    - Constant: same value across row
    - Progression: arithmetic sequence
    - XOR: combination rule
    - Distribute3: each value appears once
    
    This provides additional supervision signal and improves interpretability.
    """
    def __init__(self, num_rules: int = 4, num_attributes: int = 4):
        super().__init__()
        self.num_rules = num_rules
        self.num_attributes = num_attributes
        
    def forward(
        self,
        rule_predictions: torch.Tensor,  # (B, num_attributes, num_rules)
        rule_targets: Optional[torch.Tensor] = None  # (B, num_attributes) if available
    ) -> torch.Tensor:
        """
        Compute auxiliary rule prediction loss.
        
        If ground truth rules are not available, returns 0 (unsupervised).
        """
        if rule_targets is None:
            return torch.tensor(0.0, device=rule_predictions.device)
        
        # Cross-entropy loss for each attribute's rule prediction
        B = rule_predictions.shape[0]
        loss = 0.0
        
        for attr_idx in range(self.num_attributes):
            attr_logits = rule_predictions[:, attr_idx, :]  # (B, num_rules)
            attr_targets = rule_targets[:, attr_idx]  # (B,)
            loss += F.cross_entropy(attr_logits, attr_targets)
        
        return loss / self.num_attributes


class CombinedContrastiveLoss(nn.Module):
    """
    Combined loss function for training with all contrastive components.
    
    Total loss = CE + λ1*Contrastive + λ2*Ranking + λ3*Consistency + λ4*Auxiliary
    """
    def __init__(
        self,
        ce_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        ranking_weight: float = 0.3,
        consistency_weight: float = 0.2,
        auxiliary_weight: float = 0.1,
        temperature: float = 0.1,
        margin: float = 1.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.contrastive_weight = contrastive_weight
        self.ranking_weight = ranking_weight
        self.consistency_weight = consistency_weight
        self.auxiliary_weight = auxiliary_weight
        
        # Initialize component losses
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.ranking_loss = RankingLoss(margin=margin)
        self.consistency_loss = ConsistencyLoss()
        self.auxiliary_loss = AuxiliaryRuleLoss()
        
    def forward(
        self,
        logits: torch.Tensor,  # (B, 8) - classification logits
        targets: torch.Tensor,  # (B,) - ground truth
        context_repr: Optional[torch.Tensor] = None,  # (B, D) - for contrastive
        choice_reprs: Optional[torch.Tensor] = None,  # (B, 8, D) - for contrastive
        context_features: Optional[torch.Tensor] = None,  # (B, 8, D) - for consistency
        choice_features: Optional[torch.Tensor] = None,  # (B, 8, D) - for consistency
        rule_predictions: Optional[torch.Tensor] = None,  # for auxiliary
        rule_targets: Optional[torch.Tensor] = None  # for auxiliary
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss with all components.
        
        Returns:
            total_loss: weighted sum of all components
            loss_dict: dictionary with individual loss values for logging
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=logits.device)
        
        # 1. Cross-entropy loss (always computed)
        ce = self.ce_loss(logits, targets)
        loss_dict['ce'] = ce.item()
        total_loss = total_loss + self.ce_weight * ce
        
        # 2. Contrastive loss (if representations available)
        if context_repr is not None and choice_reprs is not None:
            contrastive = self.contrastive_loss(context_repr, choice_reprs, targets)
            loss_dict['contrastive'] = contrastive.item()
            total_loss = total_loss + self.contrastive_weight * contrastive
        
        # 3. Ranking loss
        ranking = self.ranking_loss(logits, targets)
        loss_dict['ranking'] = ranking.item()
        total_loss = total_loss + self.ranking_weight * ranking
        
        # 4. Consistency loss (if features available)
        if context_features is not None and choice_features is not None:
            consistency = self.consistency_loss(
                context_features, choice_features, targets
            )
            loss_dict['consistency'] = consistency.item()
            total_loss = total_loss + self.consistency_weight * consistency
        
        # 5. Auxiliary rule loss (if available)
        if rule_predictions is not None:
            auxiliary = self.auxiliary_loss(rule_predictions, rule_targets)
            loss_dict['auxiliary'] = auxiliary.item()
            total_loss = total_loss + self.auxiliary_weight * auxiliary
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
