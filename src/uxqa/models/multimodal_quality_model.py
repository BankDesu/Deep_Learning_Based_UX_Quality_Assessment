"""
MultiModalQualityModel — Visual + Structural gated fusion for UX quality.

Default backbone: DINOv2 ViT-S/14 (self-supervised, LVD-142M)
Fusion: learned per-sample gate α that decides how much to trust
        visual vs. structural features.

Why DINOv2?
  - Self-supervised pretraining on 142M diverse images → features transfer
    well to small fine-tuning datasets (800 UICrit train samples)
  - Patch-level spatial tokens → better UI layout understanding than
    global pooling backbones like EfficientNet
  - Linear probe with frozen backbone consistently outperforms supervised
    ImageNet models on fine-grained visual tasks

Why learned gate instead of additive correction?
  - Additive delta (old design) collapsed to ≈0 with zero-init + struct_hidden=8
  - Gate α ∈ (0,1) is bounded, interpretable, and easier to optimize
  - Gate is instance-adaptive: noisy structural features → α≈1 (rely on visual)

Supported backbones
-------------------
  dinov2_vits14   DINOv2 ViT-S/14 (timm), 384-dim  ← default
  dinov2_vitb14   DINOv2 ViT-B/14 (timm), 768-dim
  efficientnet_b4 torchvision, 1792-dim  (original, kept for comparison)
  efficientnet_v2_s  torchvision, 1280-dim
  convnextv2_tiny timm, 768-dim

Training modes
--------------
  probe    — backbone fully frozen; train heads + gate (~few hundred params)
  finetune — last N backbone blocks unfrozen
"""
from __future__ import annotations

import torch
from torch import nn

from .structural_encoder import compute_structural_features, N_STRUCT_FEATURES

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return (backbone_without_head, feature_dim)."""
    if name == "dinov2_vits14":
        import timm
        m = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )
        return m, m.num_features  # 384

    if name == "dinov2_vitb14":
        import timm
        m = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )
        return m, m.num_features  # 768

    if name == "efficientnet_b4":
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
        m = efficientnet_b4(
            weights=EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        )
        dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
        return m, dim  # 1792

    if name == "efficientnet_v2_s":
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        m = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        )
        dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
        return m, dim  # 1280

    if name == "convnextv2_tiny":
        import timm
        m = timm.create_model(
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            pretrained=pretrained,
            num_classes=0,
        )
        return m, m.num_features  # 768

    raise ValueError(
        f"Unknown backbone '{name}'. Supported: dinov2_vits14, dinov2_vitb14, "
        "efficientnet_b4, efficientnet_v2_s, convnextv2_tiny"
    )


def _freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def _unfreeze_last_n_blocks(backbone: nn.Module, name: str, n: int) -> None:
    """Unfreeze the last N blocks depending on backbone family."""
    if name.startswith("dinov2"):
        blocks = list(backbone.blocks)
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in backbone.norm.parameters():
            p.requires_grad = True

    elif name.startswith("efficientnet"):
        children = list(backbone.features.children())
        for child in children[-n:]:
            for p in child.parameters():
                p.requires_grad = True

    elif name.startswith("convnextv2"):
        stages = list(backbone.stages.children())
        for stage in stages[-n:]:
            for p in stage.parameters():
                p.requires_grad = True
        for p in backbone.head_hidden.parameters():
            p.requires_grad = True


class MultiModalQualityModel(nn.Module):
    """
    Visual + Structural gated fusion model for UX quality prediction.

    Parameters
    ----------
    backbone : str
        Backbone identifier (see module docstring for options).
    mode : "probe" | "finetune"
    dropout : float
        Dropout applied in the visual head.
    unfreeze_last_n : int
        Finetune mode: number of backbone blocks to unfreeze.
    gate_hidden : int
        Hidden dim of the gate network.
    pretrained : bool
        Load ImageNet / DINOv2 pretrained weights.
    """

    def __init__(
        self,
        backbone: str = "dinov2_vits14",
        mode: str = "probe",
        dropout: float = 0.5,
        unfreeze_last_n: int = 3,
        gate_hidden: int = 32,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone

        # ── Visual Backbone ───────────────────────────────────────────────
        self.visual_backbone, visual_dim = _build_backbone(backbone, pretrained)
        _freeze_all(self.visual_backbone)
        if mode == "finetune":
            _unfreeze_last_n_blocks(self.visual_backbone, backbone, unfreeze_last_n)

        # ── Visual Head ───────────────────────────────────────────────────
        self.visual_head = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # ── Structural Head ───────────────────────────────────────────────
        # Larger hidden than before (8→64) to actually learn layout→quality
        self.struct_head = nn.Sequential(
            nn.Linear(N_STRUCT_FEATURES, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        # ── Learned Gate ──────────────────────────────────────────────────
        # α = gate(proj(feat_v), struct_feats)  ∈ (0,1)
        # score = sigmoid(α * logit_v + (1−α) * logit_s)
        # Gate projection keeps param count reasonable regardless of visual_dim
        self.gate_proj = nn.Linear(visual_dim, gate_hidden, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(gate_hidden + N_STRUCT_FEATURES, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

        # ImageNet denormalization constants (for structural feature computation)
        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_std",  torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        )

    # ------------------------------------------------------------------

    def visual_params(self) -> list:
        return [p for p in self.visual_backbone.parameters() if p.requires_grad]

    def non_visual_params(self) -> list:
        return (
            list(self.visual_head.parameters())
            + list(self.struct_head.parameters())
            + list(self.gate_proj.parameters())
            + list(self.gate.parameters())
        )

    def _denorm(self, img: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalized [B,3,H,W] → [0,1] range."""
        return (img * self._std + self._mean).clamp(0.0, 1.0)

    # ------------------------------------------------------------------

    def forward(
        self,
        img: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        img : FloatTensor [B, 3, H, W]  (ImageNet-normalized)
        return_gate : bool
            If True, also return gate α [B, 1] for interpretability.
            α ≈ 1 → model relies on visual features.
            α ≈ 0 → model relies on structural features.

        Returns
        -------
        score : FloatTensor [B, 1]  in [0, 1]
        alpha : FloatTensor [B, 1]  (only when return_gate=True)
        """
        # Visual branch
        feat_v  = self.visual_backbone(img)      # [B, visual_dim]
        logit_v = self.visual_head(feat_v)        # [B, 1]

        # Structural branch (no gradient through pixel ops)
        img_01 = self._denorm(img)
        with torch.no_grad():
            struct_feats = compute_structural_features(img_01)   # [B, 19]
        logit_s = self.struct_head(struct_feats)                  # [B, 1]

        # Learned gate
        gv    = self.gate_proj(feat_v.detach())                   # [B, gate_hidden]
        alpha = self.gate(torch.cat([gv, struct_feats], dim=1))   # [B, 1]

        score = torch.sigmoid(alpha * logit_v + (1.0 - alpha) * logit_s)

        if return_gate:
            return score, alpha
        return score
