import torch

from uxqa.config import ModelConfig
from uxqa.models import UXAssessmentModel


def test_end_to_end_forward_shapes() -> None:
    cfg = ModelConfig(image_size=224, output_dim=64, visual_embed_dim=64, num_heads=8, ff_dim=256)
    model = UXAssessmentModel(cfg)
    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)

    out = model(x)

    assert out["ux_score"].shape == (2, 1)
    assert out["layout_quality"].shape == (2, 1)
    assert out["attention_alignment"].shape == (2, 1)
    assert out["visual_feature_map"].ndim == 4
    assert out["attention_heatmap"].ndim == 4
    assert out["layout_embedding"].shape == (2, cfg.output_dim)
    assert out["attention_embedding"].shape == (2, cfg.output_dim)


def test_explicit_pipeline_shapes() -> None:
    cfg = ModelConfig(image_size=224, output_dim=64, visual_embed_dim=64, num_heads=8, ff_dim=256)
    model = UXAssessmentModel(cfg)
    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)

    out = model.forward_pipeline(x)

    assert out["visual_feature_map"].shape[0] == 2
    assert out["visual_tokens"].shape[0] == 2
    assert out["visual_tokens"].shape[-1] == cfg.output_dim
    assert out["layout_embedding"].shape == (2, cfg.output_dim)
    assert out["layout_token"].shape == (2, 1, cfg.output_dim)
    assert out["graph_density"].shape == (2,)
    assert out["attention_heatmap"].shape[0] == 2
    assert out["attention_embedding"].shape == (2, cfg.output_dim)
    assert out["attention_token"].shape == (2, 1, cfg.output_dim)
    assert out["fused_tokens"].shape[0] == 2
    assert out["fused_tokens"].shape[-1] == cfg.output_dim
    assert out["fused_cls"].shape == (2, cfg.output_dim)
    assert out["ux_score"].shape == (2, 1)
    assert out["layout_quality"].shape == (2, 1)
    assert out["attention_alignment"].shape == (2, 1)
