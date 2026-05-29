# Adversarial Robust Real-Time Multimodal Deepfake Detection

Module 1 (Core Deepfake Detection & Mathematical Modelling) — Tasks 1–6.
Module 2 (Adversarial Robustness & Diffusion Reconstruction) is upcoming.

## Status

| # | Task | Status |
|---|---|---|
| 1 | Dataset Preparation (FF++, Celeb-DF, DFDC, FakeAVCeleb) | done |
| 2 | Temporal Multimodal Detection Module | done |
| 3 | Cross-Attention Fusion | done |
| 4 | Mathematical Modeling | done — see [`docs/math.md`](docs/math.md) |
| 5 | Real-Time Optimization (pruning + INT8 + benchmark) | done |
| 6 | Performance Evaluation (Acc / F1 / AUC / FPS / latency) | done |

## Layout

```
configs/                  YAML configs
data/
  datasets/               dataset-specific loaders (FF++, Celeb-DF, DFDC, FakeAVCeleb)
  preprocessing/          frame extraction, face crop, audio demux
  splits.py               deterministic identity-grouped train/val/test splits
  streaming.py            real-time stream simulator + sliding-window buffer
models/
  temporal_vit.py         spatial ViT + temporal Transformer (factorized attention)
  audio_encoder.py        log-mel + 1-D CNN audio encoder
  av_sync.py              audio-visual sync head (InfoNCE + sync score)
  cross_attention_fusion.py   bidirectional V<->A cross-attention fusion
  detector.py             full multimodal detector
  optimization.py         pruning, dynamic INT8 quantization, sparsity helpers
scripts/
  prepare_datasets.py     build manifests + identity-grouped splits
  extract_frames.py       face-crop frames and audio demux
  train.py                AMP training loop (cosine LR, joint detection+sync loss)
  benchmark.py            latency / FPS / realtime-factor measurement
  evaluate.py             full evaluation report (metrics + per-dataset + latency)
utils/
  video_utils.py          PyAV/OpenCV video I/O
  audio_utils.py          waveform + log-mel helpers
  metrics.py              accuracy / F1 / AUC / EER / latency summary
docs/
  math.md                 paper-ready equations for the backbone + fusion
tests/                    synthetic-data smoke tests (no datasets needed)
```

## Quickstart

```bash
pip install -r requirements.txt

# 0. Smoke tests on synthetic data (no datasets needed)
python -m tests.smoke_test
python -m tests.test_cross_attention
python -m tests.test_splits
python -m tests.test_optimization
python -m tests.test_metrics
```

## Full workflow

### 1. Build manifests (per dataset + combined splits)

```bash
python scripts/prepare_datasets.py \
    --dataset faceforensics celebdf dfdc fakeavceleb \
    --root /data/FF++ /data/Celeb-DF-v2 /data/DFDC /data/FakeAVCeleb \
    --out manifests/
```

This writes `<name>.csv` for each dataset, a `combined.csv`, and identity-grouped
`train.csv` / `val.csv` / `test.csv`. Splits are deterministic (MD5 hash with a
configurable salt) and group all clips of a given person into the same split to
prevent leakage.

### 2. Extract face crops + audio

```bash
python scripts/extract_frames.py \
    --manifest manifests/train.csv \
    --out_frames frames/ --out_audio audio/ \
    --fps 4 --max_frames 64 --crop_size 224
```

Repeat for `val.csv` and `test.csv`. The script enriches each manifest with
`frames_dir` and `audio_path` columns and writes a `*.extracted.csv`.

### 3. Train

```bash
python scripts/train.py --config configs/default.yaml
```

Trains with AMP, cosine LR + linear warmup, joint detection + AV-sync loss.
Best checkpoint (by val AUC) is saved to `checkpoints/best.pt`.

### 4. Evaluate (Task 6)

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --manifest manifests/test.extracted.csv \
    --ckpt checkpoints/best.pt \
    --out results/eval_fp32.json
```

The report contains:

* **Overall metrics**: accuracy, F1, precision, recall, AUC, AP, EER, confusion matrix
* **Per-dataset breakdown**: same metrics broken down by `dataset` column
  (FF++, Celeb-DF, DFDC, FakeAVCeleb) — exposes generalization gaps
* **Latency analysis**: per-clip mean / p50 / p95 / p99 ms, effective FPS, and
  realtime factor (`clip_seconds / latency`)

### 5. Benchmark optimized variants (Task 5)

Compare deployment variants:

```bash
# FP32 baseline
python scripts/benchmark.py --config configs/default.yaml --device cpu

# 40% global L1 pruning
python scripts/benchmark.py --config configs/default.yaml --device cpu --prune 0.4

# Dynamic INT8 quantization (CPU only)
python scripts/benchmark.py --config configs/default.yaml --device cpu --quantize

# Combined
python scripts/benchmark.py --config configs/default.yaml --device cpu \
    --prune 0.4 --quantize --out results/bench_pruned_int8.json
```

`evaluate.py` accepts the same `--prune` / `--quantize` flags, so you can score
the optimized variant on real metrics, not just synthetic timings.

## Math

Formal equations for the temporal backbone and cross-attention fusion are in
[`docs/math.md`](docs/math.md). Every symbol is cross-referenced to the
implementation file it lives in, so the doc stays in sync with the code.

## Configuration

All knobs live in [`configs/default.yaml`](configs/default.yaml). The most
common ones to change:

| Key | Meaning |
|---|---|
| `data.num_frames` | sampled frames per clip (default 16) |
| `data.frame_size` | input image side (default 224) |
| `model.embed_dim` | visual embedding dim |
| `model.spatial_depth / temporal_depth` | ViT layers |
| `model.fusion_depth / fusion_heads` | cross-attention fusion |
| `train.sync_loss_weight` | $\lambda$ in $\mathcal{L} = \mathrm{CE} + \lambda\mathcal{L}_{\mathrm{sync}}$ |

## Reference paper / repo

Stylistic inspiration: [rshaojimmy/MultiModal-DeepFake](https://github.com/rshaojimmy/MultiModal-DeepFake)
(HAMMER / DGM4 — image-text manipulation). This project addresses video+audio
deepfakes with a different architecture and dataset family.
