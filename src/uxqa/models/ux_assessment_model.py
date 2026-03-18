import torch
from torch import nn

from ..config import ModelConfig
from .attention import AttentionBranch
from .backbones import SwinTransformerEncoder
from .fusion import CrossModalTransformer
from .heads import UXPredictionHead
from .layout import LayoutBranch


class UXAssessmentModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg

        self.visual_encoder = SwinTransformerEncoder(
            in_channels=c.in_channels,
            embed_dim=c.visual_embed_dim,
            patch_size=c.patch_size,
            depth=2,
            num_heads=c.num_heads,
            ff_dim=c.ff_dim,
            dropout=c.dropout,
        )
        self.layout_branch = LayoutBranch(
            visual_dim=c.visual_embed_dim,
            out_dim=c.output_dim,
            max_elements=c.max_ui_elements,
            gnn_layers=c.gnn_layers,
        )
        self.attention_branch = AttentionBranch(
            visual_dim=c.visual_embed_dim,
            out_dim=c.output_dim,
        )
        self.visual_proj = nn.Linear(c.visual_embed_dim, c.output_dim)
        self.cross_modal_transformer = CrossModalTransformer(
            dim=c.output_dim,
            layers=c.transformer_layers,
            num_heads=c.num_heads,
            ff_dim=c.ff_dim,
            dropout=c.dropout,
        )
        self.pred_head = UXPredictionHead(c.output_dim)

    def _attention_guided_visual_tokens(
        self, visual_feature_map: torch.Tensor, attention_map: torch.Tensor
    ) -> torch.Tensor:
        weighted_map = visual_feature_map * (1.0 + attention_map)
        tokens = weighted_map.flatten(2).transpose(1, 2)
        return self.visual_proj(tokens)

    def forward(self, screenshot: torch.Tensor) -> dict[str, torch.Tensor]:
        visual_feature_map = self.visual_encoder(screenshot)
        layout = self.layout_branch(visual_feature_map)
        attention = self.attention_branch(visual_feature_map)
        visual_tokens = self._attention_guided_visual_tokens(visual_feature_map, attention.heatmap)

        fused = self.cross_modal_transformer(
            visual_tokens=visual_tokens,
            layout_token=layout.layout_token,
            attention_token=attention.attention_token,
            graph_density=layout.graph_density,
        )
        fused_cls = fused[:, 0, :]
        preds = self.pred_head(fused_cls)
        preds["visual_feature_map"] = visual_feature_map
        preds["attention_heatmap"] = attention.heatmap
        return preds
