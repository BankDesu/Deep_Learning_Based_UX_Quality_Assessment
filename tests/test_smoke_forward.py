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
