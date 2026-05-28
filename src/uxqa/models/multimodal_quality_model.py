"""
MultiModalQualityModel — Visual + Structural additive correction for UX quality.

Architecture
------------
  Visual Branch:      EfficientNet-B4 (ImageNet) → feat_v [1792] → head_v → logit_v [1]
  Structural Branch:  19 pixel-derived layout features → tiny MLP → delta [1]
  Fusion:             score = sigmoid(logit_v + delta)

The structural branch provides a learned additive correction to the visual score.
Initializing delta near 0 ensures training starts from the strong visual baseline.
This design adds only ~160 parameters beyond the visual head, preventing overfitting
on small datasets (800 training samples).

Interpretability: delta > 0 means structural features improve predicted quality;
delta < 0 means structural features reduce it.

Training modes
--------------
  probe    — backbone fully frozen; train visual head + structural correction
  finetune — last N backbone blocks unfrozen + visual head + structural correction
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from .structural_encoder import compute_structural_features, N_STRUCT_FEATURES

_VISUAL_DIM = 1792


class MultiModalQualityModel(nn.Module):
    """
    Visual-Structural additive correction model for UX quality prediction.

    Parameters
    ----------
    mode : "probe" | "finetune"
    dropout : float
        Dropout for visual head
    unfreeze_last_n : int
        Finetune mode: number of EfficientNet-B4 blocks to unfreeze
    struct_hidden : int
        Hidden dim of the structural correction MLP (kept small to avoid overfit)
    """

    def __init__(
        self,
        mode: str = "probe",
        dropout: float = 0.5,
        unfreeze_last_n: int = 3,
        struct_hidden: int = 8,
    ) -> None:
        super().__init__()

        # ── Visual Branch ─────────────────────────────────────────────────
        base = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        base.classifier = nn.Identity()
        self.visual_backbone = base

        if mode == "probe":
            for p in self.visual_backbone.parameters():
                p.requires_grad = False
        else:
            for p in self.visual_backbone.parameters():
                p.requires_grad = False
            blocks = list(self.visual_backbone.features.children())
            for block in blocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.visual_backbone.features[-1].parameters():
                p.requires_grad = True

        # Visual quality head — identical to the ImageNet baseline
        self.visual_head = nn.Sequential(
            nn.Linear(_VISUAL_DIM, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # ── Structural Correction Branch ───────────────────────────────────
        # Tiny MLP: 19 features → struct_hidden → 1 (additive logit delta)
        # Initialized near zero so training starts at the visual-only baseline.
        self.struct_correction = nn.Sequential(
            nn.Linear(N_STRUCT_FEATURES, struct_hidden),
            nn.GELU(),
            nn.Linear(struct_hidden, 1),
        )
        # Zero-init last layer → delta ≈ 0 at start
        nn.init.zeros_(self.struct_correction[-1].weight)
        nn.init.zeros_(self.struct_correction[-1].bias)

    # ------------------------------------------------------------------

    def visual_params(self) -> list:
        return [p for p in self.visual_backbone.parameters() if p.requires_grad]

    def non_visual_params(self) -> list:
        return (list(self.visual_head.parameters()) +
                list(self.struct_correction.parameters()))

    def forward(
        self,
        img: torch.Tensor,
        return_delta: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        img : FloatTensor [B, 3, H, W]  (ImageNet-normalized)
        return_delta : bool
            If True, also return structural delta [B, 1] for interpretability.

        Returns
        -------
        score : FloatTensor [B, 1]  in [0, 1]
        delta : FloatTensor [B, 1]  (only when return_delta=True)
        """
        # Visual logit
        feat_v   = self.visual_backbone(img)              # [B, 1792]
        logit_v  = self.visual_head(feat_v)               # [B, 1]

        # Structural correction (compute on denormalized image)
        mean   = img.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std    = img.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_01 = (img * std + mean).clamp(0.0, 1.0)
        with torch.no_grad():
            struct_feats = compute_structural_features(img_01)   # [B, 19]
        delta = self.struct_correction(struct_feats)              # [B, 1]

        score = torch.sigmoid(logit_v + delta)                    # [B, 1]

        if return_delta:
            return score, delta
        return score
