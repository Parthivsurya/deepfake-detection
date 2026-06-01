"""Markdown rendering for the unified robustness-evaluation ledger.

Pure formatter — takes the JSON dict produced by `scripts/robustness_eval.py`
and emits a paper-ready Markdown summary with headline numbers, per-block
tables, and a final trust-score line.

Missing blocks are simply skipped (so a partial run with `--skip_*` flags
still renders cleanly).
"""
from __future__ import annotations
from typing import Optional


def _fmt(x, ndigits: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if abs(x) >= 1e4 or (0 < abs(x) < 1e-3):
            return f"{x:.3e}"
        return f"{x:.{ndigits}f}"
    return str(x)


def _table(headers: list[str], rows: list[list]) -> str:
    sep = "|".join(["---"] * len(headers))
    lines = ["| " + " | ".join(headers) + " |",
             "|" + sep + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(c) for c in r) + " |")
    return "\n".join(lines)


def _headlines(h: dict) -> str:
    rows = [
        ["Clean accuracy", _fmt(h.get("clean_accuracy"))],
        ["Clean AUC", _fmt(h.get("clean_auc"))],
        ["Clean F1", _fmt(h.get("clean_f1"))],
        ["EER", _fmt(h.get("eer"))],
        ["Adv. accuracy (PGD @ ε_ref)", _fmt(h.get("adv_accuracy_pgd_ref"))],
        ["Adv. accuracy (FGSM @ ε_ref)", _fmt(h.get("adv_accuracy_fgsm_ref"))],
        ["Recovered accuracy (adv. @ ε_ref)", _fmt(h.get("recovered_accuracy_adv_ref"))],
        ["Mean trust score (clean)", _fmt(h.get("mean_trust_score_clean"))],
        ["Mean trust score (adv. @ ε_ref)", _fmt(h.get("mean_trust_score_adv_ref"))],
        ["Certified accuracy @ r₀", _fmt(h.get("certified_accuracy_at_r"))],
        ["Lipschitz bound (bounded subnet)",
         _fmt(h.get("lipschitz_bound_bounded_only"))],
        ["Mean certified radius (smoothing)",
         _fmt(h.get("mean_certified_radius_smoothing"))],
        ["Natural risk", _fmt(h.get("natural_risk"))],
        ["Adversarial risk @ ε_ref", _fmt(h.get("adversarial_risk_ref"))],
        ["Boundary risk @ ε_ref", _fmt(h.get("boundary_risk_ref"))],
        ["Mean latency (ms)", _fmt(h.get("mean_latency_ms"), 2)],
        ["Effective FPS", _fmt(h.get("fps_effective"), 2)],
        ["Realtime factor", _fmt(h.get("realtime_factor"), 2)],
        ["Reference ε", _fmt(h.get("reference_epsilon"))],
        ["Target r₀", _fmt(h.get("target_certified_radius"))],
    ]
    return "## Headline metrics\n\n" + _table(["Metric", "Value"], rows)


def _detection(d: Optional[dict]) -> str:
    if not d or not d.get("overall"):
        return ""
    o = d["overall"]
    rows = [[
        o.get("n"), o.get("accuracy"), o.get("f1"),
        o.get("precision"), o.get("recall"), o.get("auc"),
        o.get("ap"), o.get("eer"),
    ]]
    s = "## Detection\n\n" + _table(
        ["n", "accuracy", "f1", "precision", "recall", "auc", "ap", "eer"], rows)
    cm = o.get("confusion_matrix")
    if cm:
        m = cm["matrix"]
        s += "\n\n**Confusion matrix** (rows=true, cols=pred): "
        s += f"TN={m[0][0]} FP={m[0][1]} FN={m[1][0]} TP={m[1][1]}"
    return s


def _realtime(r: Optional[dict]) -> str:
    if not r or not r.get("mean_ms"):
        return ""
    rows = [[
        r.get("n"), r.get("mean_ms"), r.get("p50_ms"),
        r.get("p95_ms"), r.get("p99_ms"), r.get("fps_effective"),
        r.get("realtime_factor"),
    ]]
    return "## Real-time\n\n" + _table(
        ["n", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
         "fps_effective", "realtime_factor"], rows)


def _attacks(a: Optional[dict]) -> str:
    if not a:
        return ""
    s = (f"## Adversarial attacks (reference ε = "
         f"{_fmt(a.get('reference_epsilon'))})\n\n"
         "### At reference ε\n\n")
    rows = []
    for name, r in (a.get("at_reference_epsilon") or {}).items():
        rows.append([
            name, r.get("clean_accuracy"), r.get("adv_accuracy"),
            r.get("attack_success_rate"), r.get("l2_mean"), r.get("linf_mean"),
        ])
    s += _table(["attack", "clean_acc", "adv_acc", "ASR", "L2_mean", "L∞_mean"], rows)
    sweep = a.get("epsilon_sweep")
    if sweep:
        s += (f"\n\n### ε-sweep ({a.get('reference_attack')})\n\n")
        rows = [[r.get("epsilon"), r.get("clean_accuracy"),
                 r.get("adv_accuracy"), r.get("attack_success_rate")]
                for r in sweep]
        s += _table(["ε", "clean_acc", "adv_acc", "ASR"], rows)
    return s


def _recovery(rec: Optional[dict]) -> str:
    if not rec:
        return ""
    s = (f"## Forensic recovery (t★ = {rec.get('t_star')}, "
         f"recon threshold = {rec.get('recon_threshold')})\n\n")
    rows = []
    for kind in ("clean", "adversarial"):
        block = rec.get(kind) or {}
        if not block:
            continue
        ts = block.get("trust_score") or {}
        rows.append([
            kind, block.get("n_samples"), block.get("recon_rate"),
            block.get("accuracy_raw"), block.get("accuracy_recovered"),
            block.get("delta_accuracy"), ts.get("mean"),
        ])
    s += _table(
        ["input", "n", "recon_rate", "acc_raw", "acc_recovered",
         "Δacc", "trust_mean"], rows)
    if rec.get("adversarial_attack"):
        s += (f"\n\nAdversarial branch attacked with "
              f"`{rec['adversarial_attack']}` at ε = "
              f"{_fmt(rec.get('adversarial_epsilon'))}.")
    return s


def _certified(c: Optional[dict]) -> str:
    if not c:
        return ""
    s = "## Certified robustness\n\n"
    lip = (c.get("lipschitz") or {}).get("summary") or {}
    if lip:
        s += ("### Lipschitz bound (bounded sub-network)\n\n"
              "| metric | value |\n|---|---|\n"
              f"| product bound L | {_fmt(lip.get('product_bound_bounded_only'))} |\n"
              f"| n_bounded layers | {lip.get('n_bounded')} |\n"
              f"| n_unbounded (MHA) | {lip.get('n_unbounded')} |\n\n")
    curve = c.get("certified_accuracy_lipschitz")
    if curve:
        s += "### Lipschitz-margin certified-accuracy curve\n\n"
        rows = [[row.get("radius"), row.get("certified_accuracy"),
                 row.get("n_certified")] for row in curve]
        s += _table(["radius", "certified_acc", "n_certified"], rows) + "\n\n"
    sm = c.get("randomized_smoothing")
    if sm:
        s += ("### Randomized smoothing (Cohen et al.)\n\n"
              f"σ = {_fmt(sm.get('sigma'))}, n = {sm.get('n')}, "
              f"α = {_fmt(sm.get('alpha'))}\n\n"
              f"- clips evaluated: {sm.get('n_clips')}\n"
              f"- certified & correct: {sm.get('n_certified_correct')}\n"
              f"- abstained: {sm.get('n_abstain')}\n"
              f"- mean certified radius: {_fmt(sm.get('mean_certified_radius'))}\n")
    return s


def _risk(r: Optional[dict]) -> str:
    if not r:
        return ""
    s = "## Risk decomposition & accuracy–robustness trade-off\n\n"
    d = r.get("decomposition_at_reference") or {}
    if d:
        s += (f"At ε = {_fmt(r.get('reference_epsilon'))}: "
              f"R_nat = {_fmt(d.get('natural_risk'))}, "
              f"R_adv = {_fmt(d.get('adversarial_risk'))}, "
              f"R_bd = {_fmt(d.get('boundary_risk'))}.\n\n")
    tr = r.get("tradeoff_curve") or []
    if tr:
        rows = [[row.get("epsilon"), row.get("natural_risk"),
                 row.get("adversarial_risk"), row.get("boundary_risk")]
                for row in tr]
        s += _table(["ε", "R_nat", "R_adv", "R_bd"], rows)
    return s


def _continual(c: Optional[dict]) -> str:
    if not c:
        return ""
    s = (f"## Continual learning (merged from `{c.get('source')}`)\n\n"
         f"- avg final accuracy: {_fmt(c.get('avg_final_accuracy'))}\n"
         f"- backward transfer:  {_fmt(c.get('backward_transfer'))}\n"
         f"- drift triggered:    {_fmt(c.get('drift_triggered'))}\n\n")
    mat = c.get("accuracy_matrix")
    if mat:
        n_eval = len(mat[0]) if mat else 0
        headers = ["after task"] + [f"eval_{k}" for k in range(n_eval)]
        rows = []
        for t, row in enumerate(mat):
            rows.append([t] + [r.get("accuracy") for r in row])
        s += "### Accuracy matrix\n\n" + _table(headers, rows)
    return s


def render_markdown(report: dict) -> str:
    parts: list[str] = []
    parts.append("# Robustness evaluation ledger")
    parts.append("")
    parts.append(f"- config: `{report.get('config')}`")
    parts.append(f"- manifest: `{report.get('manifest')}`")
    parts.append(f"- checkpoint: `{report.get('ckpt')}`")
    parts.append(f"- device: `{report.get('device')}`")
    parts.append("")
    parts.append(_headlines(report.get("headlines") or {}))
    for fn, key in [
        (_detection, "detection"),
        (_realtime, "realtime"),
        (_attacks, "attacks"),
        (_recovery, "recovery"),
        (_certified, "certified"),
        (_risk, "risk"),
        (_continual, "continual"),
    ]:
        section = fn(report.get(key))
        if section:
            parts.append("")
            parts.append(section)
    parts.append("")
    return "\n".join(parts) + "\n"
