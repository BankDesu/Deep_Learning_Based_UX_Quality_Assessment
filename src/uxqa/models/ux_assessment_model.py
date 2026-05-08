"""
UXAssessmentModel — Layout Analysis + Rule-based pipeline (with Saliency).

Task
----
UI Quality Scoring: predict a holistic quality score correlated with human
expert ratings (UICrit dataset), with interpretable per-rule breakdown.

Pipeline
--------
  Inputs:
    screenshot   : FloatTensor [B, 3, H, W]

  Stage 1 — Visual Encoding:
    screenshot → SwinTransformerEncoder → visual_feature_map [B, C, H, W]

  Stage 2–3 — Layout Branch:
    visual_feature_map + screenshot → YOLOv8 + GNN
        → layout_token [B, 1, dim],  graph_density [B]

  Stage 4a — Attention Branch:
    visual_feature_map → SaliencyPredictor → heatmap [B, 1, H', W']
                       → AttentionMapEncoder → attention_token [B, 1, dim]

  Stage 4b — Rule Checking (algorithmic, 7 rules):
    screenshot + boxes + heatmap
        → RuleChecker → rule_scores [B, 7]

    Three rules are heatmap-enhanced (★):
        visual_balance  : saliency-weighted centroid
        cta_prominence  : attention overlap with button bboxes
        reading_flow    : top-salient pixels in upper screen

  Stage 4c — Token Fusion:
    enhanced_layout_token = layout_token + rule_token

  Stage 5 — Cross-Modal Fusion:
    visual_tokens + enhanced_layout_token + attention_token
        → CrossModalTransformer → fused_tokens

  Stage 6 — Quality Head:
    fused_cls + rule_scores
        → RuleViolationHead → { rule_scores, rule_weights, quality_score }

    quality_score = 0.5 × neural_quality + 0.5 × weighted_rule_score
    Training target: UICrit human quality rating (normalized to [0,1])

Evaluation
----------
  Primary   : Kendall's Tau τ, Spearman's ρ  (quality_score vs. UICrit rating)
  Secondary : Precision / Recall / F1 per rule (rule_scores vs. critique labels)
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .attention import AttentionBranch
from .backbones import SwinTransformerEncoder
from .fusion import CrossModalTransformer
from .heads.rule_head import RuleViolationHead
from .layout import LayoutBranch
from .rules import N_RULES, RuleChecker


class UXAssessmentModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg

        # ── Stage 1: Visual Encoder ────────────────────────────────────────
        self.visual_encoder = SwinTransformerEncoder(
            in_channels=c.in_channels,
            embed_dim=c.visual_embed_dim,
            patch_size=c.patch_size,
            depth=2,
            num_heads=c.num_heads,
            ff_dim=c.ff_dim,
            dropout=c.dropout,
        )

        # ── Stage 2–3: Layout Branch (YOLOv8 + GNN) ───────────────────────
        self.layout_branch = LayoutBranch(
            visual_dim=c.visual_embed_dim,
            out_dim=c.output_dim,
            max_elements=c.max_ui_elements,
            gnn_layers=c.gnn_layers,
            detector_backend=c.detector_backend,
            yolo_model_path=c.yolo_model_path,
            yolo_conf_threshold=c.yolo_conf_threshold,
            yolo_iou_threshold=c.yolo_iou_threshold,
        )

        # ── Stage 4a: Attention Branch ─────────────────────────────────────
        # heatmap [B,1,H',W'] enhances 3 rules (★)
        # attention_token [B,1,dim] guides visual weighting in CrossModalTransformer
        self.attention_branch = AttentionBranch(
            visual_dim=c.visual_embed_dim,
            out_dim=c.output_dim,
        )

        # ── Stage 4b: Rule Checker (7 rules, no learned parameters) ────────
        self.rule_checker = RuleChecker(max_elements=c.max_ui_elements)

        # ── Stage 4c: Rule Token Encoder ───────────────────────────────────
        self.rule_encoder = nn.Sequential(
            nn.Linear(N_RULES, c.output_dim),
            nn.GELU(),
            nn.Linear(c.output_dim, c.output_dim),
        )

        # ── Stage 5: Visual projection + Cross-Modal Fusion ───────────────
        self.visual_proj = nn.Linear(c.visual_embed_dim, c.output_dim)
        self.cross_modal_transformer = CrossModalTransformer(
            dim=c.output_dim,
            layers=c.transformer_layers,
            num_heads=c.num_heads,
            ff_dim=c.ff_dim,
            dropout=c.dropout,
            max_visual_tokens=c.max_visual_tokens,
        )

        # ── Stage 6: Quality Head ──────────────────────────────────────────
        self.rule_head = RuleViolationHead(in_dim=c.output_dim, n_rules=N_RULES)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _visual_tokens(self, visual_feature_map: torch.Tensor) -> torch.Tensor:
        """Flatten spatial dims and project to output_dim."""
        tokens = visual_feature_map.flatten(2).transpose(1, 2)  # [B, HW, C]
        return self.visual_proj(tokens)                          # [B, HW, dim]

    def _collect_boxes(
        self,
        visual_feature_map: torch.Tensor,
        screenshot: torch.Tensor,
    ) -> torch.Tensor:
        """Run detector and return padded boxes tensor [B, max_N, 4]."""
        B = screenshot.shape[0]
        device = screenshot.device
        detections = self.layout_branch.detector(visual_feature_map, screenshot=screenshot)
        max_n = max((d.boxes.shape[0] for d in detections), default=1)
        padded = torch.zeros(B, max(max_n, 1), 4, device=device)
        for i, det in enumerate(detections):
            n = det.boxes.shape[0]
            if n > 0:
                padded[i, :n] = det.boxes
        return padded

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_pipeline(
        self,
        screenshot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Full pipeline with named intermediate tensors for inspection / debugging.

        Parameters
        ----------
        screenshot : FloatTensor [B, 3, H, W]

        Returns
        -------
        visual_feature_map   [B, C, H, W]
        visual_tokens        [B, HW, dim]
        layout_embedding     [B, dim]
        layout_token         [B, 1, dim]
        graph_density        [B]
        attention_heatmap    [B, 1, H', W']
        attention_token      [B, 1, dim]
        rule_scores          [B, 7]          — 3 heatmap-enhanced (★)
        rule_token           [B, 1, dim]
        rule_weights         [B, 7]
        fused_tokens         [B, S, dim]
        fused_cls            [B, dim]
        weighted_rule_score  [B, 1]
        quality_score        [B, 1]          — primary output
        """
        # ── Stage 1 ───────────────────────────────────────────────────────
        visual_feature_map = self.visual_encoder(screenshot)

        # ── Stage 2–3 ─────────────────────────────────────────────────────
        layout = self.layout_branch(visual_feature_map, screenshot=screenshot)
        detected_boxes = self._collect_boxes(visual_feature_map, screenshot)

        # ── Stage 4a ──────────────────────────────────────────────────────
        attention = self.attention_branch(visual_feature_map)

        # ── Stage 4b ──────────────────────────────────────────────────────
        rule_scores = self.rule_checker(
            screenshot=screenshot,
            boxes=detected_boxes,
            heatmap=attention.heatmap,
        )  # [B, 7]

        # ── Stage 4c ──────────────────────────────────────────────────────
        rule_token = self.rule_encoder(rule_scores).unsqueeze(1)        # [B, 1, dim]
        enhanced_layout_token = layout.layout_token + rule_token         # [B, 1, dim]

        # ── Stage 5 ───────────────────────────────────────────────────────
        visual_tokens = self._visual_tokens(visual_feature_map)          # [B, HW, dim]

        fused_tokens = self.cross_modal_transformer(
            visual_tokens=visual_tokens,
            layout_token=enhanced_layout_token,
            attention_token=attention.attention_token,
            graph_density=layout.graph_density,
        )  # [B, S, dim]

        fused_cls = fused_tokens[:, 0, :]                                # [B, dim]

        # ── Stage 6 ───────────────────────────────────────────────────────
        head_out = self.rule_head(fused_cls, rule_scores)

        return {
            "visual_feature_map":  visual_feature_map,
            "visual_tokens":       visual_tokens,
            "layout_embedding":    layout.layout_embedding,
            "layout_token":        layout.layout_token,
            "graph_density":       layout.graph_density,
            "attention_heatmap":   attention.heatmap,
            "attention_token":     attention.attention_token,
            "rule_scores":         head_out["rule_scores"],
            "rule_token":          rule_token,
            "rule_weights":        head_out["rule_weights"],
            "fused_tokens":        fused_tokens,
            "fused_cls":           fused_cls,
            "weighted_rule_score": head_out["weighted_rule_score"],
            "quality_score":       head_out["layout_quality_score"],
        }

    def forward(
        self,
        screenshot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compact forward — primary outputs for training / inference.

        Returns
        -------
        quality_score      : FloatTensor [B, 1]        — target = UICrit rating [0,1]
        rule_scores        : FloatTensor [B, 7]        — per-rule (★ 3 heatmap-enhanced)
        rule_weights       : FloatTensor [B, 7]        — learned importance
        attention_heatmap  : FloatTensor [B, 1, H', W']
        layout_embedding   : FloatTensor [B, dim]
        visual_feature_map : FloatTensor [B, C, H, W]
        """
        pipeline = self.forward_pipeline(screenshot)
        return {
            "quality_score":       pipeline["quality_score"],
            "rule_scores":         pipeline["rule_scores"],
            "rule_weights":        pipeline["rule_weights"],
            "attention_heatmap":   pipeline["attention_heatmap"],
            "layout_embedding":    pipeline["layout_embedding"],
            "visual_feature_map":  pipeline["visual_feature_map"],
        }
