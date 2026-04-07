# Dataset Layout

This project uses the following task-to-dataset mapping:

- Layout detection: PubLayNet
- UI detection (website): WebUI Dataset
- Attention modeling: SALICON
- UX score training: WDQD
- UX evaluation benchmark: WebPage Aesthetics

Expected structure:

- data/raw/publaynet/{images,annotations}
- data/raw/webui/{images,annotations}
- data/raw/salicon/{images,fixations,maps}
- data/raw/wdqd/{images,labels}
- data/raw/webpage_aesthetics/{images,labels}
- data/manifests/*.csv
- data/interim/*
- data/processed/*

Use configs/datasets.yaml as the canonical path map for loaders.

## Download Datasets

1. Configure dataset sources in configs/dataset_sources.yaml.
2. If you use Kaggle sources, authenticate first:
	- create ~/.kaggle/kaggle.json from your Kaggle account API key
	- run: chmod 600 ~/.kaggle/kaggle.json
3. Run downloader:
	- python3 scripts/download_datasets.py

The downloader supports:

- method: kaggle (requires kaggle CLI and credentials)
- method: url (direct file download via curl)
- method: manual (placeholder, prints instruction only)
