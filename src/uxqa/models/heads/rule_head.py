"""
RuleViolationHead — learned aggregation head for rule-based UX evaluation.

Architecture
------------
  fused_cls  [B, dim]
      │
      ├─► weight_head  → learned_weights [B, N_RULES]   (softmax)
      │
      └─► quality_head → neural_quality  [B, 1]          (sigmoid)

  rule_scores [B, N_RULES]  (from RuleChecker — algorithmic, not learned)

  Output:
      layout_quality_score = 0.5 * neural_quality
                           + 0.5 * Σ(rule_scores * learned_weights)

  This hybrid design keeps rule scores interpretable while allowing the
  model to learn which rules matter most for overall layout quality.
"""
from __future__ import annotations

import torch
from torch import nn

from ..rules.definitions import N_RULES, RULE_NAMES, RULE_WEIGHTS


class RuleViolationHead(nn.Module):
    """
    Parameters
    ----------
    in_dim  : int   — dimension of fused_cls token from CrossModalTransformer
    n_rules : int   — number of UX rules (default N_RULES = 9)
    """

    def __init__(self, in_dim: int, n_rules: int = N_RULES) -> None:
        super().__init__()
        self.n_rules = n_rules

        # Learned rule importance weights (soft-attention over rule scores)
        self.weight_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, n_rules),
            nn.Softmax(dim=-1),
        )

        # Complementary neural quality estimate
        self.quality_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Prior weights as a non-trainable buffer (for logging / analysis)
        self.register_buffer(
            "prior_weights",
            torch.tensor(RULE_WEIGHTS, dtype=torch.float32),
        )

    def forward(
        self,
        fused_cls: torch.Tensor,    # [B, in_dim]
        rule_scores: torch.Tensor,  # [B, N_RULES]
    ) -> dict[str, torch.Tensor]:
        """
        Returns
        -------
        rule_scores         : FloatTensor [B, N_RULES] — raw algorithmic scores (pass-through)
        rule_weights        : FloatTensor [B, N_RULES] — learned importance weights
        weighted_rule_score : FloatTensor [B, 1]       — Σ(rule_scores * learned_weights)
        layout_quality_score: FloatTensor [B, 1]       — final hybrid quality score
        """
        learned_weights = self.weight_head(fused_cls)           # [B, N_RULES]
        neural_quality  = self.quality_head(fused_cls)           # [B, 1]
        weighted_score  = (rule_scores * learned_weights).sum(dim=-1, keepdim=True)  # [B, 1]

        # Hybrid: 90% learned neural quality, 10% weighted rule score
        # (rule_scores has no learnable params; keep its influence small so the
        # quality signal is dominated by the trainable head.)
        layout_quality = 0.9 * neural_quality + 0.1 * weighted_score

        return {
            "rule_scores":          rule_scores,      # [B, N_RULES] — interpretable
            "rule_weights":         learned_weights,  # [B, N_RULES] — what model learned
            "weighted_rule_score":  weighted_score,   # [B, 1]
            "layout_quality_score": layout_quality,   # [B, 1] — final output
        }

    @property
    def rule_names(self) -> list[str]:
        return RULE_NAMES
