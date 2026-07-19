"""Extract attribution embeddings and per-clip predictions; plot t-SNE.

Loads `attribution_best.pt`, runs a labeled attribution manifest through the
model, and writes:

    <out_prefix>_scores.csv       clip_id, generator_id, pred_id, per-class softmax
    <out_prefix>_embeddings.npz   embeds (N, D), labels (N,), clip_ids (N,)
    <out_prefix>_tsne.png         2-D t-SNE coloured by generator class

Model hyper-parameters are taken from the config stored inside the checkpoint,
so the script cannot silently build a mismatched architecture.

The manifest may lack `frames_dir` / `audio_path` columns (the CSVs stored in
the attribution-extracted-frames Kaggle dataset don't have them); they are
derived as <frame_root>/<clip_id> and <audio_root>/<clip_id>.wav. Clips whose
frames directory is missing are dropped with a warning. Missing audio is fine:
the model gates silent clips (has_audio=0), matching training behaviour for
Celeb-DF.

Usage:
    python scripts/attribution_embeddings.py \
        --ckpt checkpoints/attribution_best.pt \
        --manifest manifests/attribution_val.csv \
        --out-prefix results/attribution_val
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from attribution.dataset import AttributionDataset  # noqa: E402
from attribution.generators import GENERATOR_REGISTRY, num_known_classes  # noqa: E402
from attribution.model import SourceAttributionModel  # noqa: E402
from data.datasets.base import VideoManifest  # noqa: E402


def build_model(cfg: dict) -> SourceAttributionModel:
    # mirrors scripts/train_attribution.py::build_model
    m = cfg["model"]
    return SourceAttributionModel(
        image_size=cfg["data"]["frame_size"],
        patch_size=m["patch_size"],
        embed_dim=m["embed_dim"],
        spatial_depth=m["spatial_depth"],
        temporal_depth=m["temporal_depth"],
        num_heads=m["num_heads"],
        mlp_ratio=m["mlp_ratio"],
        dropout=m["dropout"],
        max_frames=max(cfg["data"]["num_frames"], 64),
        spectral_dim=m.get("spectral_dim", 256),
        residual_dim=m.get("residual_dim", 256),
        head_hidden=m.get("head_hidden", 384),
        num_classes=num_known_classes(),
        use_audio=m.get("use_audio", True),
        audio_sample_rate=cfg["data"].get("audio_sample_rate", 16000),
        audio_embed_dim=m.get("audio_embed_dim", 256),
        audio_fp_dim=m.get("audio_fp_dim", 256),
        audio_n_mels=m.get("audio_n_mels", 80),
        audio_encoder_kind=m.get("audio_encoder", "cnn"),
        wav2vec_pretrained=m.get("wav2vec_pretrained", "facebook/wav2vec2-base"),
        wav2vec_freeze=m.get("wav2vec_freeze", True),
        use_physio=m.get("use_physio", False),
        physio_embed_dim=m.get("physio_embed_dim", 128),
        physio_fps=m.get("physio_fps", 4.0),
        backbone=m.get("backbone", "temporal_vit"),
    )


def prepare_manifest(path: str, frame_root: str, audio_root: str) -> VideoManifest:
    m = VideoManifest.load(path)
    df = m.df
    if "frames_dir" not in df.columns:
        df["frames_dir"] = df["clip_id"].map(lambda c: str(Path(frame_root) / str(c)))
    if "audio_path" not in df.columns:
        def _audio(c):
            p = Path(audio_root) / f"{c}.wav"
            return str(p) if p.is_file() else ""
        df["audio_path"] = df["clip_id"].map(_audio)
    have = df["frames_dir"].map(lambda p: Path(str(p)).is_dir())
    if (~have).any():
        print(f"[warn] dropping {(~have).sum()}/{len(df)} clips with no frames dir")
        m.df = df[have].reset_index(drop=True)
    n_audio = (m.df["audio_path"] != "").sum()
    print(f"[manifest] {len(m.df)} clips, {n_audio} with audio")
    return m


@torch.no_grad()
def run(model, loader, device):
    model.eval()
    logits, embeds, labels, clip_ids = [], [], [], []
    for i, batch in enumerate(loader):
        frames = batch["frames"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True) if model.use_audio else None
        has_audio = (batch["has_audio"].to(device, non_blocking=True)
                     if model.use_audio else None)
        out = model(frames, waveform=audio, has_audio=has_audio)
        logits.append(out["logits"].cpu())
        embeds.append(out["embed"].cpu())
        labels.append(batch["generator_id"])
        clip_ids.extend(batch["clip_id"])
        if i % 20 == 0:
            print(f"  batch {i}/{len(loader)}")
    return (torch.cat(logits), torch.cat(embeds),
            torch.cat(labels), clip_ids)


def plot_tsne(embeds: np.ndarray, labels: np.ndarray, out_png: Path,
              perplexity: float, title: str) -> None:
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xy = TSNE(n_components=2, perplexity=perplexity, init="pca",
              random_state=42).fit_transform(embeds)
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    cmap = plt.get_cmap("tab10")
    for k, gid in enumerate(sorted(set(labels.tolist()))):
        info = GENERATOR_REGISTRY.get(int(gid))
        name = info.name if info else f"class {gid}"
        mask = labels == gid
        ax.scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=0.7,
                   color=cmap(k % 10), label=f"{name} (n={mask.sum()})")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--frame-root", default="frames")
    p.add_argument("--audio-root", default="audio")
    p.add_argument("--out-prefix", default="results/attribution")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", default=default_device)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--no-tsne", action="store_true")
    args = p.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    print(f"[ckpt] backbone={cfg['model'].get('backbone')} "
          f"epoch={state.get('epoch')} metrics={state.get('metrics', {}).get('f1_macro')}")
    model = build_model(cfg)
    model.load_state_dict(state["model"])
    model = model.to(args.device)

    manifest = prepare_manifest(args.manifest, args.frame_root, args.audio_root)
    ds = AttributionDataset(
        manifest, training=False,
        num_frames=cfg["data"]["num_frames"],
        frame_size=cfg["data"]["frame_size"],
        audio_sample_rate=cfg["data"].get("audio_sample_rate", 16000),
        audio_seconds=cfg["data"].get("audio_seconds", 4.0),
        load_audio=cfg["model"].get("use_audio", True),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    logits, embeds, labels, clip_ids = run(model, loader, args.device)
    probs = F.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(axis=1)
    y = labels.numpy()

    from sklearn.metrics import accuracy_score, classification_report, f1_score
    present = sorted(set(y.tolist()))
    names = [GENERATOR_REGISTRY[c].name if c in GENERATOR_REGISTRY else str(c)
             for c in present]
    print(f"\naccuracy={accuracy_score(y, preds):.4f} "
          f"macro-F1={f1_score(y, preds, labels=present, average='macro'):.4f}")
    print(classification_report(y, preds, labels=present, target_names=names,
                                zero_division=0))

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    scores = pd.DataFrame({"clip_id": clip_ids, "generator_id": y, "pred_id": preds})
    for j, c in enumerate(range(probs.shape[1])):
        info = GENERATOR_REGISTRY.get(c)
        scores[f"p_{info.name if info else c}"] = probs[:, j]
    scores.to_csv(f"{prefix}_scores.csv", index=False)
    print(f"wrote {prefix}_scores.csv")

    np.savez_compressed(f"{prefix}_embeddings.npz",
                        embeds=embeds.numpy(), labels=y,
                        clip_ids=np.array(clip_ids))
    print(f"wrote {prefix}_embeddings.npz")

    if not args.no_tsne:
        plot_tsne(embeds.numpy(), y, Path(f"{prefix}_tsne.png"),
                  args.perplexity,
                  f"Attribution embeddings t-SNE — {Path(args.manifest).stem} "
                  f"(n={len(y)})")


if __name__ == "__main__":
    main()
