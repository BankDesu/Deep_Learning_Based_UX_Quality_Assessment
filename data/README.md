# Dataset Layout

This project uses **UICrit + RICO** for UX quality score training.

## Datasets

| Dataset | Role | Source |
|---------|------|--------|
| UICrit | Annotations + quality ratings | https://github.com/google-research-datasets/uicrit |
| RICO | Mobile UI screenshots | Kaggle / interactionmining.org/rico |

## Expected structure

```
data/
├── raw/
│   ├── uicrit/
│   │   └── uicrit_public.csv     ← cloned from GitHub
│   └── rico/
│       └── combined/
│           └── <rico_id>.jpg     ← ~1,000 screenshots needed
├── manifests/
│   ├── uicrit_train.csv
│   ├── uicrit_val.csv
│   └── uicrit_test.csv
└── interim/
    └── ux_score_training/
        ├── ux_score_training_train.jsonl
        ├── ux_score_training_val.jsonl
        └── ux_score_training_test.jsonl
```

## Download

### UICrit (annotations)
```bash
git clone --depth 1 https://github.com/google-research-datasets/uicrit data/raw/uicrit
```

### RICO (screenshots)
Requires Kaggle credentials (`~/.kaggle/kaggle.json`):
```bash
kaggle datasets download -d <slug> -p data/raw/rico --unzip
```

See `configs/dataset_sources.yaml` for the Kaggle slug once configured.
