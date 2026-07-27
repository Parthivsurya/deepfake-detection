# Kaggle session — TRE reliability ablation (with vs without)

Goal tonight: train **two** models that are identical except the reliability
mechanism, then evaluate both to get the "with reliability vs without" numbers
(TPR/FPR, F1/precision/recall, AUC) for the A\* figures.

- `configs/trust_off.yaml` → **without reliability** (baseline)
- `configs/trust_on.yaml` → **with reliability** (TRE + TSF)

Both use the same backbone / data / seed / epochs. Code is on `main`
(commit `51562e0`). **Prepend `%cd /kaggle/working/code` to every cell.**

> ⚠️ T4×2 note: DataParallel auto-engages. Watch CPU RAM — if it climbs toward
> 30 GiB, stop and resume with `--resume checkpoints/<dir>/best.pt` (both scripts
> support it).

---

## Cell 1 — bootstrap (clone updated code + deps)
```python
import os, subprocess
os.chdir("/kaggle/working")
if not os.path.exists("code"):
    subprocess.run(["git","clone","https://github.com/Parthivsurya/deepfake-detection.git","code"], check=True)
os.chdir("/kaggle/working/code")
subprocess.run(["git","pull"])                      # ensure TRE commit 51562e0 is present
subprocess.run(["pip","install","-q","-r","requirements.txt"])
subprocess.run(["pip","install","-q","--no-deps","facenet-pytorch"])
print("HEAD:", subprocess.run(["git","rev-parse","--short","HEAD"], capture_output=True, text=True).stdout.strip())
```

## Cell 2 — DATA PREP (reuse your Session-7 Celeb-DF setup)
Point the configs at the **same frames/manifests** your Session-7 EfficientNet-B0
run used (the one that gave Celeb-DF AUC 0.709), so the ablation is comparable.
Paste your known-good Celeb-DF frame-extraction + manifest cells here (symlink
past the `celeb-df-v2` double-nesting, extract frames, build
`manifests/{train,val}.extracted.csv`). Do **not** change the data between the
two runs.

## Cell 3 — sanity: confirm the trust toggle loads
```python
%cd /kaggle/working/code
import yaml
for c in ["configs/trust_off.yaml","configs/trust_on.yaml"]:
    m = yaml.safe_load(open(c))["model"]
    print(c, "-> use_trust =", m.get("use_trust"))
```

## Cell 4 — train WITHOUT reliability (baseline)
```python
%cd /kaggle/working/code
import yaml
cfg = yaml.safe_load(open("configs/trust_off.yaml"))
# match your data manifests here if they differ:
# cfg["data"]["manifest_train"] = "manifests/train.extracted.csv"
# cfg["data"]["manifest_val"]   = "manifests/val.extracted.csv"
cfg["train"]["epochs"] = 20
yaml.safe_dump(cfg, open("configs/_trust_off_run.yaml","w"))
!python scripts/train.py --config configs/_trust_off_run.yaml --device cuda
# resume if the kernel restarts:
# !python scripts/train.py --config configs/_trust_off_run.yaml --device cuda --resume checkpoints/trust_off/best.pt
```

## Cell 5 — train WITH reliability (TRE + TSF)
```python
%cd /kaggle/working/code
import yaml
cfg = yaml.safe_load(open("configs/trust_on.yaml"))
# cfg["data"]["manifest_train"] = "manifests/train.extracted.csv"
# cfg["data"]["manifest_val"]   = "manifests/val.extracted.csv"
cfg["train"]["epochs"] = 20
yaml.safe_dump(cfg, open("configs/_trust_on_run.yaml","w"))
!python scripts/train.py --config configs/_trust_on_run.yaml --device cuda
# resume: --resume checkpoints/trust_on/best.pt
```
The `[e.. s..]` log line now also reflects L_trust inside `loss`. Best model is
`checkpoints/trust_on/best.pt`.

## Cell 6 — evaluate BOTH, save per-clip scores
```python
%cd /kaggle/working/code
import subprocess, os
os.makedirs("results/trust_ablation", exist_ok=True)
VAL_MANIFEST = "manifests/val.extracted.csv"   # same split used for the baseline
for tag in ["trust_off","trust_on"]:
    subprocess.run([
        "python","scripts/evaluate.py",
        "--config",   f"configs/{tag}.yaml",
        "--manifest", VAL_MANIFEST,
        "--ckpt",     f"checkpoints/{tag}/best.pt",
        "--device",   "cuda",
        "--save_scores", f"results/trust_ablation/scores_{tag}.csv",
        "--out",         f"results/trust_ablation/eval_{tag}.json",
    ], check=True)
    print("done", tag)
```
Flags verified against `scripts/evaluate.py` (`--config --manifest --ckpt --device
--save_scores --out`). `evaluate.py` prefers the model cfg embedded in the
checkpoint, so the trust params reconstruct automatically.

## Cell 7 — push checkpoints + scores to the artifacts dataset
```python
%cd /kaggle/working/code
import json, os, shutil, subprocess
stage = "/kaggle/working/upload"; os.makedirs(stage, exist_ok=True)
# keep existing files: download what we want to preserve first (see workflow note),
# or push these to a NEW small dataset to avoid clobbering dfdc-smoke-artifacts.
for tag in ["trust_off","trust_on"]:
    shutil.copy(f"checkpoints/{tag}/best.pt", f"{stage}/best_{tag}.pt")
shutil.copytree("results/trust_ablation", f"{stage}/trust_ablation", dirs_exist_ok=True)
json.dump({"title":"trust-ablation","id":"parthivsuryakb/trust-ablation",
           "licenses":[{"name":"CC0-1.0"}]}, open(f"{stage}/dataset-metadata.json","w"))
subprocess.run(["kaggle","datasets","create","-p",stage])   # first time; later: version -p stage -m "..."
```

---

## After the run — download `results/trust_ablation/` locally, then I:
1. compute TPR/FPR, F1/precision/recall at threshold 0.5 (and the ROC) from the
   two `scores_*.csv` files,
2. build the **reliability vs non-reliability** figures (Tasks 1 & 2),
3. add the **AUC** row for our model to the SOTA comparison (Task 3),
4. wire the real numbers into `paper.tex`.

All from measured per-clip scores — nothing fabricated.
