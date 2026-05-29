# Adversarial Robust Real-Time Multimodal Deepfake Detection

Module 1 — Tasks 1 & 2 implementation:
1. **Dataset Preparation** — loaders for FaceForensics++, Celeb-DF, DFDC, FakeAVCeleb; train/val/test splits; frame extraction; streaming simulation.
2. **Temporal Multimodal Detection Module** — Temporal Vision Transformer (frame sequence + temporal attention) with audio encoder and audio-visual synchronization head.

## Layout
```
configs/                 YAML configs
data/
  datasets/              dataset-specific loaders (FF++, Celeb-DF, DFDC, FakeAVCeleb)
  preprocessing/         frame extraction, face crop, audio extraction
  splits.py              deterministic train/val/test splits
  streaming.py           real-time stream simulator
models/
  temporal_vit.py        Temporal Vision Transformer
  audio_encoder.py       1D audio CNN encoder
  av_sync.py             audio-visual sync head
  detector.py            full multimodal detector
scripts/
  prepare_datasets.py    build manifests + run frame extraction
  extract_frames.py      standalone frame extractor
  train.py               training loop
utils/                   shared helpers
tests/                   synthetic-data smoke tests
```

## Quickstart
```bash
pip install -r requirements.txt

# 1. Smoke test on synthetic data (no datasets required)
python -m tests.smoke_test

# 2. Build a manifest for one of the supported datasets
python scripts/prepare_datasets.py \
    --dataset faceforensics \
    --root /path/to/FaceForensics++ \
    --out manifests/ff.csv

# 3. Extract face crops
python scripts/extract_frames.py \
    --manifest manifests/ff.csv \
    --out frames/ff \
    --fps 4

# 4. Train
python scripts/train.py --config configs/default.yaml
```

See `configs/default.yaml` for all knobs.
