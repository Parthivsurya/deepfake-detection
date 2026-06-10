"""Continual / adaptive fine-tuning CLI (Module 2 Task 4).

Walks a sequence of manifests (the "task stream" — each one a different
deepfake family or dataset) and fine-tunes the detector on each in turn
while protecting prior knowledge with experience replay and EWC.

After every task the script evaluates accuracy on every prior task so the
JSON report contains the full forgetting matrix you need to compute
average-accuracy and backward-transfer metrics.

Example:
    python scripts/continual_train.py \\
        --config configs/default.yaml \\
        --ckpt checkpoints/best.pt \\
        --tasks manifests/ffpp.csv manifests/celebdf.csv manifests/dfdc.csv \\
        --epochs 1 --replay_lambda 1.0 --ewc_lambda 1000.0 \\
        --replay_capacity 512 \\
        --out results/continual.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.datasets import VideoManifest, VideoClipDataset  # noqa: E402
from models import MultimodalDeepfakeDetector  # noqa: E402
from continual import (  # noqa: E402
    ClassBalancedReplayBuffer,
    ElasticWeightConsolidation,
    ContinualTrainer,
    ContinualConfig,
    DriftDetector,
)


def build_model(cfg: dict) -> MultimodalDeepfakeDetector:
    m, d = cfg["model"], cfg["data"]
    return MultimodalDeepfakeDetector(
        image_size=d["frame_size"],
        patch_size=m["patch_size"],
        embed_dim=m["embed_dim"],
        spatial_depth=m["spatial_depth"],
        temporal_depth=m["temporal_depth"],
        num_heads=m["num_heads"],
        mlp_ratio=m["mlp_ratio"],
        dropout=0.0,
        max_frames=max(d["num_frames"], 64),
        audio_sample_rate=d["audio_sample_rate"],
        audio_embed_dim=m["audio_embed_dim"],
        fusion_dim=m["fusion_dim"],
        fusion_depth=m.get("fusion_depth", 2),
        fusion_heads=m.get("fusion_heads", 8),
        max_audio_tokens=m.get("max_audio_tokens", 256),
    )


def _collate(batch):
    return torch.utils.data.default_collate(
        [{k: v for k, v in b.items() if k != "clip_id"} for b in batch])


def _build_loader(cfg, manifest_path: str, batch_size: int, num_workers: int,
                  training: bool):
    manifest = VideoManifest.load(manifest_path)
    ds = VideoClipDataset(
        manifest,
        num_frames=cfg["data"]["num_frames"],
        frame_size=cfg["data"]["frame_size"],
        audio_sample_rate=cfg["data"]["audio_sample_rate"],
        audio_seconds=cfg["data"]["audio_seconds"],
        training=training,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=training,
                      num_workers=num_workers, pin_memory=True,
                      collate_fn=_collate)


@torch.inference_mode()
def _evaluate(model, loader, device: str, max_batches=None) -> dict:
    model.eval()
    n, correct = 0, 0
    fakeprob_sum = 0.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        has_audio = batch.get("has_audio")
        if has_audio is not None:
            has_audio = has_audio.to(device)
        out = model(batch["frames"].to(device), batch["audio"].to(device),
                    has_audio=has_audio)
        prob_fake = torch.softmax(out["logits"], dim=-1)[:, 1]
        pred = (prob_fake > 0.5).long()
        correct += int((pred == batch["label"].to(device)).sum().item())
        fakeprob_sum += float(prob_fake.sum().item())
        n += int(batch["label"].size(0))
    return {"accuracy": correct / max(n, 1),
            "p_fake_mean": fakeprob_sum / max(n, 1),
            "n_samples": n}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tasks", nargs="+", required=True,
                   help="ordered list of manifest csvs, one per task")
    p.add_argument("--eval_tasks", nargs="+", default=None,
                   help="eval manifests (defaults to --tasks)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--replay_lambda", type=float, default=1.0)
    p.add_argument("--replay_batch_size", type=int, default=4)
    p.add_argument("--replay_capacity", type=int, default=256,
                   help="exemplars stored per class")
    p.add_argument("--ewc_lambda", type=float, default=0.0)
    p.add_argument("--consolidation_batches", type=int, default=8)
    p.add_argument("--max_eval_batches", type=int, default=None)
    p.add_argument("--drift_window", type=int, default=256)
    p.add_argument("--drift_psi_threshold", type=float, default=0.25)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    eval_paths = args.eval_tasks or args.tasks

    model = build_model(cfg).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    buffer = ClassBalancedReplayBuffer(capacity_per_class=args.replay_capacity)
    ewc = ElasticWeightConsolidation(lam=1.0)
    trainer = ContinualTrainer(
        model=model, optimizer=optimizer, buffer=buffer, ewc=ewc,
        config=ContinualConfig(
            replay_lambda=args.replay_lambda,
            replay_batch_size=args.replay_batch_size,
            ewc_lambda=args.ewc_lambda,
            consolidation_batches=args.consolidation_batches,
        ),
        device=args.device,
    )
    drift = DriftDetector(window_size=args.drift_window,
                           psi_threshold=args.drift_psi_threshold)

    accuracy_matrix = []  # acc_matrix[t][k] = acc on task k after training task t
    train_logs = []

    for t, train_path in enumerate(args.tasks):
        train_loader = _build_loader(cfg, train_path, args.batch_size,
                                       args.num_workers, training=True)
        log = trainer.train_on(train_loader, n_epochs=args.epochs)
        log["task_index"] = t
        log["task_manifest"] = train_path
        train_logs.append(log)

        # consolidate EWC on the same data we just trained on
        if args.ewc_lambda > 0:
            consol_loader = _build_loader(cfg, train_path, args.batch_size,
                                           args.num_workers, training=False)
            trainer.finish_task(consol_loader)

        # eval on all tasks seen so far + future ones (full matrix for forgetting/forward-transfer)
        row = []
        for k, eval_path in enumerate(eval_paths):
            eval_loader = _build_loader(cfg, eval_path, args.eval_batch_size,
                                         args.num_workers, training=False)
            metrics = _evaluate(model, eval_loader, args.device, args.max_eval_batches)
            metrics["task_index"] = k
            metrics["task_manifest"] = eval_path
            row.append(metrics)
            # feed prediction scores into drift detector while we have them
            for batch in eval_loader:
                with torch.inference_mode():
                    has_audio = batch.get("has_audio")
                    if has_audio is not None:
                        has_audio = has_audio.to(args.device)
                    out = model(batch["frames"].to(args.device),
                                batch["audio"].to(args.device),
                                has_audio=has_audio)
                    p_fake = torch.softmax(out["logits"], dim=-1)[:, 1]
                drift.update(p_fake)
                break  # one batch of drift signal per task is enough
        accuracy_matrix.append(row)

    # Forgetting metrics (Lopez-Paz & Ranzato style):
    # Backward transfer = mean(acc[T-1][k] - acc[k][k]) over k < T-1
    if len(accuracy_matrix) > 1:
        T = len(accuracy_matrix)
        bwt_terms = []
        for k in range(T - 1):
            bwt_terms.append(accuracy_matrix[-1][k]["accuracy"]
                             - accuracy_matrix[k][k]["accuracy"])
        backward_transfer = sum(bwt_terms) / len(bwt_terms)
        avg_final_accuracy = sum(accuracy_matrix[-1][k]["accuracy"]
                                  for k in range(T)) / T
    else:
        backward_transfer = 0.0
        avg_final_accuracy = (accuracy_matrix[-1][0]["accuracy"]
                              if accuracy_matrix else 0.0)

    report = {
        "config": args.config,
        "ckpt": args.ckpt,
        "tasks": args.tasks,
        "eval_tasks": eval_paths,
        "device": args.device,
        "replay_lambda": args.replay_lambda,
        "replay_capacity_per_class": args.replay_capacity,
        "ewc_lambda": args.ewc_lambda,
        "train_logs": train_logs,
        "accuracy_matrix": accuracy_matrix,
        "backward_transfer": backward_transfer,
        "avg_final_accuracy": avg_final_accuracy,
        "drift": drift.stats(),
        "drift_triggered": drift.is_drifting(),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
