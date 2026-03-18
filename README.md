# Multi-Modal UX Quality Assessment (Scaffold)

This repository is now structured to match your architecture diagram:

1. `UI Screenshot -> Swin Transformer Encoder -> Visual Feature Map`
2. `Layout Branch`: YOLOv8 adapter -> layout graph -> GNN encoder -> layout token
3. `Attention Branch`: saliency predictor -> attention encoder -> attention token
4. `Cross-Modal Transformer`: layout-aware + attention-guided fusion
5. `UX Prediction Head`: UX score, layout quality, attention alignment

## Folder Structure

```text
configs/
  default.yaml
scripts/
  train.py
  infer.py
src/
  uxqa/
    config.py
    models/
      backbones/
        swin_encoder.py
      layout/
        detector.py
        graph_builder.py
        gnn_encoder.py
        branch.py
      attention/
        saliency.py
        encoder.py
        branch.py
      fusion/
        cross_modal_transformer.py
      heads/
        ux_head.py
      ux_assessment_model.py
tests/
  test_smoke_forward.py
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run smoke test:

```bash
python -m pytest
```

Run training stub:

```bash
python scripts/train.py
```

Run inference stub:

```bash
python scripts/infer.py
```

## Notes

- The current YOLOv8 and saliency components are adapter scaffolds with stable interfaces.
- You can plug in production detectors/predictors later without changing the fusion and head wiring.
