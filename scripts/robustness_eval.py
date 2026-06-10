"""Unified robustness evaluation ledger (Module 2 Task 6).

The capstone of Module 2: ties together Tasks 1–5 into a single end-to-end
report combining clean detection, real-time benchmarks, adversarial accuracy,
forensic recovery, certified radius, and risk decomposition. Produces both a
JSON ledger (machine-readable) and a Markdown summary (paper-ready).

Blocks in the report:

  detection    — clean accuracy, F1, AUC, EER, confusion matrix
                 (delegates to `utils.metrics.compute_metrics`)
  realtime     — per-clip latency, FPS, realtime factor (Task 5)
  attacks      — clean vs adversarial accuracy under FGSM / PGD / CW / DeepFool
                 at a reference epsilon, plus an epsilon sweep for the
                 reference attack (Tasks 1, 2)
  recovery     — diffusion purification on clean inputs and (optionally) on
                 adversarial inputs: accuracy_raw vs accuracy_recovered, mean
                 trust score, recon rate, p_orig/p_recon/p_final summaries
                 (Task 3)
  certified    — Lipschitz product bound over the bounded sub-network,
                 margin-based certified-accuracy curve, randomized smoothing
                 on a small subset (Task 5)
  risk         — natural / adversarial / boundary risk and the
                 accuracy–robustness trade-off curve over the same epsilon
                 sweep (Task 5)
  continual    — (optional) merged forgetting matrix from a prior
                 `continual_train.py` run (Task 4)

Headline numbers (also surfaced at the top of the markdown report):
  clean_accuracy, adv_accuracy@ref_eps, recovered_accuracy@ref_eps,
  certified_accuracy@r0, mean_trust_score, mean_latency_ms, fps_effective,
  realtime_factor.

Example:
    python scripts/robustness_eval.py \\
        --config configs/default.yaml \\
        --manifest manifests/test.extracted.csv \\
        --ckpt checkpoints/best.pt \\
        --attack pgd --epsilons 0.005 0.01 0.02 0.03 0.05 \\
        --reference_epsilon 0.03 \\
        --sigma 0.25 --max_smoothing_clips 8 \\
        --continual_report results/continual.json \\
        --out_json results/robustness_eval.json \\
        --out_md  results/robustness_eval.md
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.datasets import VideoManifest, VideoClipDataset  # noqa: E402
from models import MultimodalDeepfakeDetector  # noqa: E402
from adversarial import (  # noqa: E402
    build_attack,
    evaluate_all_attacks,
    sweep_epsilon,
)
from diffusion import (  # noqa: E402
    DiffusionSchedule,
    SmallUNet,
    DDPM,
    HeuristicPerturbationDetector,
    ForensicRecoveryPipeline,
)
from robustness import (  # noqa: E402
    LipschitzEstimator,
    SmoothedClassifier,
    ABSTAIN,
    compute_margins,
    certified_accuracy_curve,
    margin_summary,
    natural_risk,
    adversarial_risk,
    risk_decomposition,
    accuracy_robustness_tradeoff,
    infer_conv_input_shapes,
)
from utils import compute_metrics, equal_error_rate, latency_summary  # noqa: E402
from utils.report import render_markdown  # noqa: E402


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

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
    out = torch.utils.data.default_collate(
        [{k: v for k, v in b.items() if k != "clip_id"} for b in batch])
    out["clip_id"] = [b["clip_id"] for b in batch]
    return out


def _build_loader(cfg, manifest_path, batch_size, num_workers):
    manifest = VideoManifest.load(manifest_path)
    ds = VideoClipDataset(
        manifest,
        num_frames=cfg["data"]["num_frames"],
        frame_size=cfg["data"]["frame_size"],
        audio_sample_rate=cfg["data"]["audio_sample_rate"],
        audio_seconds=cfg["data"]["audio_seconds"],
        training=False,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True,
                       collate_fn=_collate)


# ---------------------------------------------------------------------------
# Block 1 — clean detection + per-clip latency
# ---------------------------------------------------------------------------

def _run_detection(model, loader, device, threshold) -> dict:
    """Clean detection metrics + per-clip latency, in one pass."""
    model.eval()
    cuda = device.startswith("cuda")
    probs, labels = [], []
    latencies_ms: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True)
            audio = batch["audio"].to(device, non_blocking=True)
            has_audio = batch["has_audio"].to(device, non_blocking=True)
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(frames, audio, has_audio=has_audio)
            if cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.extend([(t1 - t0) * 1000.0 / frames.size(0)] * frames.size(0))
            p = F.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
            probs.append(p)
            labels.append(batch["label"].cpu().numpy())
    probs = np.concatenate(probs) if probs else np.array([])
    labels = np.concatenate(labels) if labels else np.array([])
    detection = compute_metrics(labels, probs, threshold=threshold)
    detection["eer"] = equal_error_rate(labels, probs)
    return {"detection": detection, "latencies_ms": latencies_ms}


# ---------------------------------------------------------------------------
# Block 4 — forensic recovery (clean + optionally adversarial)
# ---------------------------------------------------------------------------

def _build_pipeline(model, cfg_args, device) -> ForensicRecoveryPipeline:
    schedule = DiffusionSchedule(T=cfg_args.diffusion_T,
                                  schedule=cfg_args.diffusion_schedule)
    eps_net = SmallUNet(in_channels=3, base_ch=cfg_args.unet_base_ch)
    if cfg_args.diffusion_ckpt:
        st = torch.load(cfg_args.diffusion_ckpt, map_location=device, weights_only=False)
        eps_net.load_state_dict(st["model"] if "model" in st else st)
    diffusion = DDPM(eps_net=eps_net, schedule=schedule).to(device)
    pert_detector = HeuristicPerturbationDetector().to(device)
    return ForensicRecoveryPipeline(
        detector=model,
        perturbation_detector=pert_detector,
        diffusion=diffusion,
        recon_threshold=cfg_args.recon_threshold,
        t_star=cfg_args.t_star,
    )


def _calibrate_perturbation_detector(pipeline, loader, device, max_batches, k):
    clean = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        clean.append(batch["frames"].to(device))
    if clean:
        pipeline.perturbation_detector.calibrate(torch.cat(clean, dim=0), k=k)


def _run_recovery(pipeline, loader, device, attack, max_batches) -> dict:
    """Walk the loader through the pipeline; optionally attack first."""
    trust_all, p_orig_all, p_recon_all, p_final_all, labels_all = [], [], [], [], []
    n_recon, n_total = 0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        frames = batch["frames"].to(device)
        audio = batch["audio"].to(device)
        has_audio = batch.get("has_audio")
        if has_audio is not None:
            has_audio = has_audio.to(device)
        labels = batch["label"].to(device)
        if attack is not None:
            res = attack.perturb(frames, audio, has_audio, labels)
            frames = res.frames_adv
        out = pipeline.run(frames, audio, has_audio=has_audio)
        trust_all.append(out.trust_score.cpu())
        p_orig_all.append(out.p_orig_fake.cpu())
        p_recon_all.append(out.p_recon_fake.cpu())
        p_final_all.append(out.p_final_fake.cpu())
        labels_all.append(labels.cpu())
        n_recon += int((out.p_perturbation > pipeline.recon_threshold).sum().item())
        n_total += int(frames.size(0))
    trust = torch.cat(trust_all) if trust_all else torch.zeros(0)
    p_orig = torch.cat(p_orig_all) if p_orig_all else torch.zeros(0)
    p_recon = torch.cat(p_recon_all) if p_recon_all else torch.zeros(0)
    p_final = torch.cat(p_final_all) if p_final_all else torch.zeros(0)
    labels = torch.cat(labels_all) if labels_all else torch.zeros(0, dtype=torch.long)
    if n_total == 0:
        return {"n_samples": 0}
    acc_raw = ((p_orig > 0.5).long() == labels).float().mean().item()
    acc_rec = ((p_final > 0.5).long() == labels).float().mean().item()
    return {
        "n_samples": n_total,
        "n_reconstructed": n_recon,
        "recon_rate": n_recon / n_total,
        "accuracy_raw": acc_raw,
        "accuracy_recovered": acc_rec,
        "delta_accuracy": acc_rec - acc_raw,
        "trust_score": {
            "mean": float(trust.mean()), "min": float(trust.min()),
            "max": float(trust.max()),
        },
        "p_orig_fake_mean": float(p_orig.mean()),
        "p_recon_fake_mean": float(p_recon.mean()),
        "p_final_fake_mean": float(p_final.mean()),
    }


# ---------------------------------------------------------------------------
# Block 5 — certified radius (Lipschitz + randomized smoothing)
# ---------------------------------------------------------------------------

def _run_certified(model, loader, args, device) -> dict:
    sample_batch = next(iter(loader))
    sf = sample_batch["frames"][:1].to(device)
    sa = sample_batch["audio"][:1].to(device)
    sh = (sample_batch["has_audio"][:1].to(device)
          if sample_batch.get("has_audio") is not None else None)
    conv_shapes = infer_conv_input_shapes(
        model, lambda m: m(sf, sa, has_audio=sh)
    )
    lip = LipschitzEstimator().estimate(
        model, input_shape_by_conv=conv_shapes,
        n_iter=args.lipschitz_power_iters,
    )
    margins_out = compute_margins(model, loader, device=device,
                                   max_batches=args.max_margin_batches)
    L = lip.product_bound
    ca_curve = certified_accuracy_curve(
        margins=margins_out["margins"],
        lipschitz_constant=L,
        correct=margins_out["correct"],
        radii=args.radii,
    )
    # randomized smoothing on a small subset (expensive)
    smoothed = SmoothedClassifier(base=model, sigma=args.sigma, num_classes=2)
    smoothing_results: list[dict] = []
    n_done = 0
    for batch in loader:
        if n_done >= args.max_smoothing_clips:
            break
        for i in range(batch["frames"].size(0)):
            if n_done >= args.max_smoothing_clips:
                break
            f = batch["frames"][i].to(device)
            a = batch["audio"][i].to(device)
            ha = (batch["has_audio"][i].to(device).unsqueeze(0)
                  if batch.get("has_audio") is not None else None)
            y = int(batch["label"][i].item())
            c_hat, R = smoothed.certify(
                f, a, has_audio=ha,
                n0=args.n0_smoothing, n=args.n_smoothing_samples,
                batch_size=args.smoothing_batch_size, alpha=args.smoothing_alpha,
            )
            smoothing_results.append({
                "label": y, "predicted": c_hat,
                "certified_radius": R, "abstain": c_hat == ABSTAIN,
                "correct_and_certified": (c_hat == y),
            })
            n_done += 1
    if smoothing_results:
        n_abst = sum(1 for r in smoothing_results if r["abstain"])
        n_cert = sum(1 for r in smoothing_results
                     if not r["abstain"] and r["correct_and_certified"])
        avg_R = (sum(r["certified_radius"] for r in smoothing_results
                     if not r["abstain"])
                 / max(len(smoothing_results) - n_abst, 1))
    else:
        n_abst = n_cert = 0
        avg_R = 0.0
    return {
        "lipschitz": {"summary": lip.summary(), "per_layer": lip.per_layer},
        "margins": margin_summary(margins_out["margins"]),
        "certified_accuracy_lipschitz": ca_curve,
        "randomized_smoothing": {
            "sigma": args.sigma,
            "n0": args.n0_smoothing,
            "n": args.n_smoothing_samples,
            "alpha": args.smoothing_alpha,
            "n_clips": len(smoothing_results),
            "n_certified_correct": n_cert,
            "n_abstain": n_abst,
            "mean_certified_radius": avg_R,
            "samples": smoothing_results,
        },
    }


# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------

def _ca_at(curve: list, target_r: float) -> Optional[float]:
    """Pick the CA value at the largest radius ≤ target_r."""
    eligible = [row for row in curve if row["radius"] <= target_r + 1e-12]
    if not eligible:
        return None
    return float(eligible[-1]["certified_accuracy"])


def _headlines(report: dict, ref_eps: float, target_r: float) -> dict:
    h: dict = {}
    det = report.get("detection", {}).get("overall")
    if det:
        h["clean_accuracy"] = det.get("accuracy")
        h["clean_auc"] = det.get("auc")
        h["clean_f1"] = det.get("f1")
        h["eer"] = det.get("eer")
    rt = report.get("realtime")
    if rt:
        h["mean_latency_ms"] = rt.get("mean_ms")
        h["fps_effective"] = rt.get("fps_effective")
        h["realtime_factor"] = rt.get("realtime_factor")
    atks = report.get("attacks", {}).get("at_reference_epsilon", {})
    if atks:
        h["adv_accuracy_pgd_ref"] = atks.get("pgd", {}).get("adv_accuracy")
        h["adv_accuracy_fgsm_ref"] = atks.get("fgsm", {}).get("adv_accuracy")
    rec = report.get("recovery", {})
    if rec.get("adversarial"):
        h["recovered_accuracy_adv_ref"] = rec["adversarial"].get("accuracy_recovered")
        h["mean_trust_score_adv_ref"] = (rec["adversarial"].get("trust_score") or {}).get("mean")
    if rec.get("clean"):
        h["mean_trust_score_clean"] = (rec["clean"].get("trust_score") or {}).get("mean")
    cert = report.get("certified", {})
    if cert.get("certified_accuracy_lipschitz"):
        h["certified_accuracy_at_r"] = _ca_at(cert["certified_accuracy_lipschitz"], target_r)
        h["lipschitz_bound_bounded_only"] = (
            (cert.get("lipschitz") or {}).get("summary", {}).get("product_bound_bounded_only"))
    if cert.get("randomized_smoothing"):
        h["mean_certified_radius_smoothing"] = cert["randomized_smoothing"].get("mean_certified_radius")
    risk = report.get("risk")
    if risk:
        decomp = risk.get("decomposition_at_reference")
        if decomp:
            h["natural_risk"] = decomp.get("natural_risk")
            h["adversarial_risk_ref"] = decomp.get("adversarial_risk")
            h["boundary_risk_ref"] = decomp.get("boundary_risk")
    h["reference_epsilon"] = ref_eps
    h["target_certified_radius"] = target_r
    return h


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--ckpt", required=True)
    default_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--device", default=default_device)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max_batches", type=int, default=None,
                   help="cap batches per block (defensible for quick smoke runs)")
    # block toggles
    p.add_argument("--skip_attacks", action="store_true")
    p.add_argument("--skip_recovery", action="store_true")
    p.add_argument("--skip_certified", action="store_true")
    p.add_argument("--skip_risk", action="store_true")
    # attacks
    p.add_argument("--attack", default="pgd", choices=["fgsm", "pgd", "cw", "deepfool"])
    p.add_argument("--epsilons", nargs="+", type=float,
                   default=[0.005, 0.01, 0.02, 0.03, 0.05])
    p.add_argument("--reference_epsilon", type=float, default=0.03)
    p.add_argument("--pgd_steps", type=int, default=10)
    p.add_argument("--cw_steps", type=int, default=30)
    p.add_argument("--deepfool_steps", type=int, default=20)
    # recovery
    p.add_argument("--diffusion_ckpt", default=None)
    p.add_argument("--diffusion_T", type=int, default=1000)
    p.add_argument("--diffusion_schedule", default="cosine",
                   choices=["linear", "cosine"])
    p.add_argument("--unet_base_ch", type=int, default=32)
    p.add_argument("--t_star", type=int, default=50)
    p.add_argument("--recon_threshold", type=float, default=0.5)
    p.add_argument("--calibrate_k", type=float, default=2.0)
    p.add_argument("--calibrate_batches", type=int, default=2)
    p.add_argument("--recovery_attack_adv", default="pgd",
                   choices=["fgsm", "pgd", "cw", "deepfool"],
                   help="attack used to stress-test the recovery pipeline")
    # certified
    p.add_argument("--lipschitz_power_iters", type=int, default=30)
    p.add_argument("--radii", nargs="+", type=float,
                   default=[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--target_radius", type=float, default=0.05,
                   help="r used for the headline 'certified accuracy at radius' number")
    p.add_argument("--max_margin_batches", type=int, default=None)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--n0_smoothing", type=int, default=50)
    p.add_argument("--n_smoothing_samples", type=int, default=500)
    p.add_argument("--smoothing_batch_size", type=int, default=16)
    p.add_argument("--smoothing_alpha", type=float, default=0.001)
    p.add_argument("--max_smoothing_clips", type=int, default=8)
    # continual merge
    p.add_argument("--continual_report", default=None,
                   help="path to a prior continual_train.py JSON to merge in")
    # output
    p.add_argument("--out_json", default=None)
    p.add_argument("--out_md", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    loader = _build_loader(cfg, args.manifest, args.batch_size, args.num_workers)

    model = build_model(cfg).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    report: dict = {
        "config": args.config,
        "manifest": args.manifest,
        "ckpt": args.ckpt,
        "device": args.device,
        "reference_epsilon": args.reference_epsilon,
        "target_certified_radius": args.target_radius,
    }

    # ----- Block 1: detection + latency ----------------------------------
    det = _run_detection(model, loader, args.device, args.threshold)
    report["detection"] = {"overall": det["detection"]}
    lat = latency_summary(det["latencies_ms"])
    T_frames = cfg["data"]["num_frames"]
    if lat.get("mean_ms"):
        lat["fps_effective"] = T_frames / (lat["mean_ms"] / 1000.0)
        lat["clips_per_sec"] = 1000.0 / lat["mean_ms"]
        lat["realtime_factor"] = (T_frames / 4.0) / (lat["mean_ms"] / 1000.0)
    report["realtime"] = lat

    # ----- Block 2: adversarial attacks ----------------------------------
    if not args.skip_attacks:
        all_atk = evaluate_all_attacks(
            model, loader, device=args.device, epsilon=args.reference_epsilon,
            pgd_steps=args.pgd_steps, cw_steps=args.cw_steps,
            deepfool_steps=args.deepfool_steps, max_batches=args.max_batches,
        )
        # epsilon sweep for the chosen reference attack family
        attack_kwargs = ({"steps": args.pgd_steps}
                        if args.attack == "pgd" else {})
        sweep = sweep_epsilon(
            model, loader, args.attack, args.epsilons,
            device=args.device, attack_kwargs=attack_kwargs,
            max_batches=args.max_batches,
        )
        report["attacks"] = {
            "reference_attack": args.attack,
            "reference_epsilon": args.reference_epsilon,
            "at_reference_epsilon": all_atk,
            "epsilon_sweep": sweep,
        }

    # ----- Block 3: forensic recovery (clean + adversarial) --------------
    if not args.skip_recovery:
        pipeline = _build_pipeline(model, args, args.device)
        _calibrate_perturbation_detector(pipeline, loader, args.device,
                                          args.calibrate_batches, args.calibrate_k)
        rec_clean = _run_recovery(pipeline, loader, args.device, None,
                                    args.max_batches)
        kw = ({"alpha": args.reference_epsilon / 4, "steps": args.pgd_steps}
              if args.recovery_attack_adv == "pgd" else {})
        adv_atk = build_attack(args.recovery_attack_adv, model,
                                epsilon=args.reference_epsilon, **kw)
        rec_adv = _run_recovery(pipeline, loader, args.device, adv_atk,
                                  args.max_batches)
        report["recovery"] = {
            "t_star": args.t_star,
            "recon_threshold": args.recon_threshold,
            "clean": rec_clean,
            "adversarial": rec_adv,
            "adversarial_attack": args.recovery_attack_adv,
            "adversarial_epsilon": args.reference_epsilon,
        }

    # ----- Block 4: certified radius -------------------------------------
    if not args.skip_certified:
        report["certified"] = _run_certified(model, loader, args, args.device)

    # ----- Block 5: risk decomposition + trade-off -----------------------
    if not args.skip_risk:
        nat = natural_risk(model, loader, device=args.device,
                            max_batches=args.max_batches)
        kw = ({"alpha": args.reference_epsilon / 4, "steps": args.pgd_steps}
              if args.attack == "pgd" else {})
        ref_atk = build_attack(args.attack, model,
                                epsilon=args.reference_epsilon, **kw)
        adv = adversarial_risk(model, loader, ref_atk, device=args.device,
                                max_batches=args.max_batches)
        decomp = risk_decomposition(nat, adv)

        def _factory(eps: float):
            kw_eps = ({"alpha": eps / 4, "steps": args.pgd_steps}
                      if args.attack == "pgd" else {})
            return build_attack(args.attack, model, epsilon=eps, **kw_eps)

        tradeoff = accuracy_robustness_tradeoff(
            model, loader, _factory, args.epsilons,
            device=args.device, max_batches=args.max_batches,
        )
        report["risk"] = {
            "reference_attack": args.attack,
            "reference_epsilon": args.reference_epsilon,
            "decomposition_at_reference": decomp,
            "tradeoff_curve": tradeoff,
        }

    # ----- Block 6 (optional): merge external continual report -----------
    if args.continual_report:
        ext = json.loads(Path(args.continual_report).read_text())
        report["continual"] = {
            "source": args.continual_report,
            "tasks": ext.get("tasks"),
            "accuracy_matrix": ext.get("accuracy_matrix"),
            "backward_transfer": ext.get("backward_transfer"),
            "avg_final_accuracy": ext.get("avg_final_accuracy"),
            "drift_triggered": ext.get("drift_triggered"),
        }

    # ----- Headlines -----------------------------------------------------
    report["headlines"] = _headlines(report, args.reference_epsilon,
                                       args.target_radius)

    print(json.dumps(report["headlines"], indent=2))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_md, "w") as f:
            f.write(render_markdown(report))


if __name__ == "__main__":
    main()
