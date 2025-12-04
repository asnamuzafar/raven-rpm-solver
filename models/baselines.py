"""
Stage D: Baseline Models and Hybrid Variants

Multiple alternative approaches for benchmarking:
1. CNN-Direct: No relational reasoning, just classify
2. Neural Relation Network (RN): Pairwise relations
3. Symbolic Rule-Based Reasoner: Explicit logical rules
4. Hybrid Model: Combines DL and symbolic reasoning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class CNNDirectBaseline(nn.Module):
    """
    Simplest baseline: Concatenate all features and classify.
    No explicit relational reasoning.
    """
    def __init__(
        self, 
        feature_dim: int = 512, 
        hidden_dim: int = 512, 
        num_choices: int = 8
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 16, hidden_dim),  # 8 context + 8 choices
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_choices)
        )
        
    def forward(
        self, 
        context_features: torch.Tensor, 
        choice_features: torch.Tensor
    ) -> torch.Tensor:
        B = context_features.shape[0]
        # Simply concatenate all features
        all_features = torch.cat([context_features, choice_features], dim=1)  # (B, 16, 512)
        all_features = all_features.view(B, -1)  # (B, 16*512)
        return self.classifier(all_features)  # (B, 8)


class RelationNetwork(nn.Module):
    """
    Relation Network: Process pairs of panels to infer relations.
    Classic architecture for relational reasoning tasks.
    """
    def __init__(
        self, 
        feature_dim: int = 512, 
        hidden_dim: int = 256, 
        num_choices: int = 8
    ):
        super().__init__()
        self.num_choices = num_choices
        
        # Process pairs of panels
        self.g_theta = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Aggregate and score
        self.f_phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self, 
        context_features: torch.Tensor, 
        choice_features: torch.Tensor
    ) -> torch.Tensor:
        B = context_features.shape[0]
        scores = []
        
        for c in range(self.num_choices):
            choice = choice_features[:, c:c+1, :]  # (B, 1, 512)
            # Complete puzzle: 8 context + 1 choice = 9 panels
            panels = torch.cat([context_features, choice], dim=1)  # (B, 9, 512)
            
            # Compute all pairwise relations
            relations = []
            for i in range(9):
                for j in range(9):
                    if i != j:
                        pair = torch.cat([panels[:, i], panels[:, j]], dim=1)  # (B, 1024)
                        rel = self.g_theta(pair)  # (B, hidden)
                        relations.append(rel)
            
            # Aggregate relations
            relations = torch.stack(relations, dim=1)  # (B, 72, hidden)
            aggregated = relations.mean(dim=1)  # (B, hidden)
            
            # Score this choice
            score = self.f_phi(aggregated)  # (B, 1)
            scores.append(score)
        
        return torch.cat(scores, dim=1)  # (B, 8)


class SymbolicReasoner:
    """
    Rule-based reasoner using symbolic attributes.
    Applies explicit logical rules as specified in goal.md:
    - Constant: Same value across row/column
    - Progression: Arithmetic sequence (e.g., +1, +2, +3)
    - XOR: Combination rule between elements  
    - Distribution: Each value appears once per row
    
    Provides transparent rule traces for interpretability.
    Note: This is not an nn.Module as it doesn't have learnable parameters.
    """
    def __init__(self):
        self.rules = ['constant', 'progression', 'xor', 'distribute_three']
        self.rule_traces = []  # Store rule traces for explanation
        
    def check_constant(self, row_attrs: List) -> Tuple[bool, str]:
        """Check if attribute is constant across row"""
        is_constant = len(set(row_attrs)) == 1
        trace = f"Values {row_attrs}: {'CONSTANT' if is_constant else 'not constant'}"
        return is_constant, trace
    
    def check_progression(self, row_attrs: List) -> Tuple[bool, str]:
        """Check if attribute follows arithmetic progression"""
        if len(row_attrs) != 3:
            return False, "Need 3 values for progression"
        try:
            vals = [int(x) if isinstance(x, str) and x.isdigit() else hash(x) % 10 
                    for x in row_attrs]
            diff1 = vals[1] - vals[0]
            diff2 = vals[2] - vals[1]
            is_prog = diff1 == diff2
            trace = f"Values {row_attrs} -> diffs [{diff1}, {diff2}]: {'PROGRESSION' if is_prog else 'not progression'}"
            return is_prog, trace
        except:
            return False, "Error computing progression"
    
    def check_xor(self, row_attrs: List) -> Tuple[bool, str]:
        """Check XOR pattern (combination of two elements produces third)"""
        if len(row_attrs) != 3:
            return False, "Need 3 values for XOR"
        # Simplified XOR check based on attribute values
        try:
            vals = set(row_attrs)
            is_xor = len(vals) == 3 or (len(vals) == 1)  # All different or all same
            trace = f"Values {row_attrs}: {'XOR pattern' if is_xor else 'not XOR'}"
            return is_xor, trace
        except:
            return False, "Error checking XOR"
    
    def check_distribute_three(self, row_attrs: List) -> Tuple[bool, str]:
        """Check if each value appears exactly once in row (distribution)"""
        is_dist = len(set(row_attrs)) == len(row_attrs)
        trace = f"Values {row_attrs}: {'DISTRIBUTE' if is_dist else 'not distributed'} (unique: {len(set(row_attrs))})"
        return is_dist, trace
    
    def infer_rule(
        self, 
        context_attrs: List[Dict], 
        attr_name: str
    ) -> Tuple[List, List, List[str]]:
        """
        Infer which rule applies for a given attribute across rows.
        Returns detected rules, partial row, and rule traces.
        """
        # Extract attribute values for each row
        # Grid: 0,1,2 (row0), 3,4,5 (row1), 6,7,? (row2)
        row0 = [context_attrs[i][attr_name] for i in [0, 1, 2]]
        row1 = [context_attrs[i][attr_name] for i in [3, 4, 5]]
        row2_partial = [context_attrs[i][attr_name] for i in [6, 7]]
        
        rules_detected = []
        traces = [f"Analyzing attribute '{attr_name}':"]
        traces.append(f"  Row 0: {row0}")
        traces.append(f"  Row 1: {row1}")
        traces.append(f"  Row 2 (partial): {row2_partial}")
        
        # Check constant rule
        const0, t0 = self.check_constant(row0)
        const1, t1 = self.check_constant(row1)
        if const0 and const1:
            rules_detected.append(('constant', row0[0]))
            traces.append(f"  ✓ CONSTANT rule detected: value={row0[0]}")
        
        # Check progression rule
        prog0, t0 = self.check_progression(row0)
        prog1, t1 = self.check_progression(row1)
        if prog0 and prog1:
            rules_detected.append(('progression', None))
            traces.append(f"  ✓ PROGRESSION rule detected")
        
        # Check XOR rule
        xor0, t0 = self.check_xor(row0)
        xor1, t1 = self.check_xor(row1)
        if xor0 and xor1:
            rules_detected.append(('xor', set(row0)))
            traces.append(f"  ✓ XOR rule detected")
        
        # Check distribution rule
        dist0, t0 = self.check_distribute_three(row0)
        dist1, t1 = self.check_distribute_three(row1)
        if dist0 and dist1:
            rules_detected.append(('distribute_three', set(row0)))
            traces.append(f"  ✓ DISTRIBUTE rule detected: values={set(row0)}")
        
        if not rules_detected:
            traces.append(f"  ? No clear rule detected")
            
        return rules_detected, row2_partial, traces
    
    def predict_missing(
        self, 
        context_attrs: List[Dict], 
        attr_name: str, 
        rules: List, 
        row2_partial: List
    ) -> Tuple[List, List[str]]:
        """Predict the missing attribute value based on detected rules"""
        predictions = []
        traces = []
        
        for rule_name, rule_data in rules:
            if rule_name == 'constant':
                predictions.append(rule_data)
                traces.append(f"  CONSTANT: missing value should be '{rule_data}'")
            elif rule_name == 'progression':
                try:
                    vals = [hash(x) % 10 for x in row2_partial]
                    diff = vals[1] - vals[0] if len(vals) >= 2 else 0
                    pred_val = vals[-1] + diff
                    predictions.append(pred_val)
                    traces.append(f"  PROGRESSION: {row2_partial} + diff={diff} -> {pred_val}")
                except:
                    pass
            elif rule_name == 'xor':
                # For XOR, predict based on combination
                if rule_data and len(row2_partial) >= 2:
                    used = set(row2_partial)
                    remaining = rule_data - used
                    if remaining:
                        pred = list(remaining)[0]
                        predictions.append(pred)
                        traces.append(f"  XOR: missing from {rule_data} given {row2_partial} -> '{pred}'")
            elif rule_name == 'distribute_three':
                used = set(row2_partial)
                remaining = rule_data - used
                if remaining:
                    pred = list(remaining)[0]
                    predictions.append(pred)
                    traces.append(f"  DISTRIBUTE: missing from {rule_data} given {row2_partial} -> '{pred}'")
                    
        return predictions, traces
    
    def score_choice(
        self, 
        context_attrs: List[Dict], 
        choice_attrs: List[Dict], 
        choice_idx: int
    ) -> Tuple[int, List[str]]:
        """Score how well a choice completes the puzzle with detailed trace"""
        score = 0
        explanations = []
        
        for attr_name in ['shape', 'size', 'color', 'count']:
            rules, row2_partial, rule_traces = self.infer_rule(context_attrs, attr_name)
            predictions, pred_traces = self.predict_missing(context_attrs, attr_name, rules, row2_partial)
            
            choice_val = choice_attrs[choice_idx][attr_name]
            
            # Check if choice matches any predicted value
            matched = False
            for pred in predictions:
                if pred == choice_val:
                    score += 1
                    matched = True
                    explanations.extend(rule_traces)
                    explanations.extend(pred_traces)
                    explanations.append(f"  ✓ Choice {choice_idx} {attr_name}='{choice_val}' MATCHES prediction")
                    break
            
            if not matched and rules:
                explanations.append(f"  ✗ Choice {choice_idx} {attr_name}='{choice_val}' does not match")
                    
        return score, explanations
    
    def predict(
        self, 
        context_attrs: List[Dict], 
        choice_attrs: List[Dict]
    ) -> Tuple[int, List[int], List[str]]:
        """
        Predict the best choice based on symbolic rules.
        
        Args:
            context_attrs: list of 8 dicts with symbolic attributes
            choice_attrs: list of 8 dicts with symbolic attributes
        Returns:
            best_choice: int 0-7
            scores: list of scores for each choice
            explanation: list of rule traces
        """
        scores = []
        all_explanations = []
        
        for i in range(8):
            score, expl = self.score_choice(context_attrs, choice_attrs, i)
            scores.append(score)
            all_explanations.append(expl)
        
        best_choice = max(range(8), key=lambda i: scores[i])
        return best_choice, scores, all_explanations[best_choice]
    
    def get_full_trace(
        self, 
        context_attrs: List[Dict], 
        choice_attrs: List[Dict]
    ) -> Dict:
        """
        Get complete rule trace for all choices.
        Required by goal.md for interpretability/explanation quality.
        """
        best_choice, scores, _ = self.predict(context_attrs, choice_attrs)
        
        full_trace = {
            'best_choice': best_choice,
            'scores': scores,
            'choice_explanations': {},
            'detected_rules': {}
        }
        
        # Get detected rules for each attribute
        for attr_name in ['shape', 'size', 'color', 'count']:
            rules, _, traces = self.infer_rule(context_attrs, attr_name)
            full_trace['detected_rules'][attr_name] = {
                'rules': [(r[0], str(r[1]) if r[1] else None) for r in rules],
                'traces': traces
            }
        
        # Get explanation for each choice
        for i in range(8):
            score, expl = self.score_choice(context_attrs, choice_attrs, i)
            full_trace['choice_explanations'][i] = {
                'score': score,
                'is_best': i == best_choice,
                'explanation': expl
            }
        
        return full_trace


class HybridReasoner(nn.Module):
    """
    Hybrid model combining deep learning and symbolic reasoning.
    DL proposes candidates, symbolic verifies/refines.
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        # Deep learning component
        self.dl_reasoner = nn.Sequential(
            nn.Linear(feature_dim * 9, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Learnable weight for combining DL and symbolic scores
        self.symbolic_weight = nn.Parameter(torch.tensor(0.5))
        
        # Symbolic reasoner (non-learnable)
        self.symbolic = SymbolicReasoner()
        
    def forward(
        self, 
        context_features: torch.Tensor, 
        choice_features: torch.Tensor,
        context_attrs: Optional[List[List[Dict]]] = None, 
        choice_attrs: Optional[List[List[Dict]]] = None
    ) -> torch.Tensor:
        """
        Combines DL scores with symbolic scores.
        
        Args:
            context_features: (B, 8, 512)
            choice_features: (B, 8, 512)
            context_attrs: Optional list of symbolic attributes for context
            choice_attrs: Optional list of symbolic attributes for choices
        """
        B = context_features.shape[0]
        
        # DL scores
        dl_scores = []
        for i in range(8):
            choice = choice_features[:, i, :]
            combined = torch.cat([context_features.view(B, -1), choice], dim=1)
            score = self.dl_reasoner(combined)
            dl_scores.append(score)
        dl_scores = torch.cat(dl_scores, dim=1)  # (B, 8)
        dl_scores = F.softmax(dl_scores, dim=1)
        
        # If symbolic attributes provided, compute symbolic scores
        if context_attrs is not None and choice_attrs is not None:
            symbolic_scores = torch.zeros_like(dl_scores)
            for b in range(B):
                for i in range(8):
                    score, _ = self.symbolic.score_choice(
                        context_attrs[b], choice_attrs[b], i
                    )
                    symbolic_scores[b, i] = score
            symbolic_scores = F.softmax(symbolic_scores, dim=1)
            
            # Weighted combination
            w = torch.sigmoid(self.symbolic_weight)
            combined_scores = w * dl_scores + (1 - w) * symbolic_scores
            return combined_scores
        
        return dl_scores
    
    def explain(
        self, 
        context_attrs: List[Dict], 
        choice_attrs: List[Dict], 
        predicted_idx: int
    ) -> Dict:
        """Get symbolic explanation for prediction"""
        _, scores, explanation = self.symbolic.predict(context_attrs, choice_attrs)
        return {
            'predicted': predicted_idx,
            'symbolic_scores': scores,
            'explanation': explanation
        }

