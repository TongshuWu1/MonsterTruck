# ===========================
# Cell 9 — TIMING-FOCUSED evaluation (what you care most about)
#
# Primary metrics per method:
#   - prediction time per step   (pred_time_step)
#   - update time per step       (train_time_step)
#   - rebuild time per step      (rebuild_time_step)
#   - wall time per step         (wall_time_step = pred + train + rebuild)
#   - average wall time (overall mean + running average curve)
#
# Also reports (very useful):
#   - update-only train time distribution (train_time_step where update_flag_step==1)
#   - update frequency (fraction of steps with update_flag==1)
#   - wall-time-to-success (optional) if you early-stop on success (episodes end at success)
#
# Assumes Cells 6/7/8 logged run_traces fields:
#   pred_time_step, train_time_step, rebuild_time_step, wall_time_step, update_flag_step
# ===========================

import numpy as np
import matplotlib.pyplot as plt

if "EVAL_REGISTRY" not in globals() or len(EVAL_REGISTRY) == 0:
    raise RuntimeError("EVAL_REGISTRY is empty. Run Cells 6/7/8 first.")

METHOD_ORDER = ["PALSGP", "OSGPR_GLOBAL", "SVGP_GLOBAL", "EXACTGP_GLOBAL"]
METHODS = [m for m in METHOD_ORDER if m in EVAL_REGISTRY] + [m for m in EVAL_REGISTRY.keys() if m not in METHOD_ORDER]
METHODS = list(dict.fromkeys(METHODS))

print("Methods found:", METHODS)

def _cat(method, key):
    arrs = []
    for tr in EVAL_REGISTRY[method].get("run_traces", []):
        if key in tr and tr[key] is not None:
            a = np.asarray(tr[key], dtype=np.float64).reshape(-1)
            if a.size:
                arrs.append(a)
    return np.concatenate(arrs) if len(arrs) else np.array([], dtype=np.float64)

def _running_avg(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    return np.cumsum(x) / (np.arange(x.size) + 1.0)

def _ms(x):
    return 1e3 * float(x)

# ------------------------------------------------------------
# Build timing summary per method
# ------------------------------------------------------------
summ = {}
for m in METHODS:
    pred = _cat(m, "pred_time_step")
    trn  = _cat(m, "train_time_step")
    reb  = _cat(m, "rebuild_time_step")
    wall = _cat(m, "wall_time_step")
    upd  = _cat(m, "update_flag_step")

    # Align lengths safely
    n = min(pred.size if pred.size else wall.size,
            trn.size if trn.size else wall.size,
            reb.size if reb.size else wall.size,
            wall.size)
    if n <= 0:
        summ[m] = dict(n_steps=0)
        continue

    pred = pred[:n] if pred.size else np.zeros((n,), dtype=np.float64)
    trn  = trn[:n]  if trn.size  else np.zeros((n,), dtype=np.float64)
    reb  = reb[:n]  if reb.size  else np.zeros((n,), dtype=np.float64)
    wall = wall[:n] if wall.size else (pred + trn + reb)
    upd  = upd[:n]  if upd.size  else np.zeros((n,), dtype=np.float64)

    upd_mask = upd >= 0.5
    trn_upd = trn[upd_mask] if np.any(upd_mask) else np.array([], dtype=np.float64)
    wall_upd = wall[upd_mask] if np.any(upd_mask) else np.array([], dtype=np.float64)

    summ[m] = dict(
        n_steps=n,
        pred=pred, trn=trn, reb=reb, wall=wall, upd=upd,
        upd_freq=float(np.mean(upd_mask)) if n else np.nan,

        pred_mean=float(np.mean(pred)),
        pred_p50=float(np.median(pred)),
        pred_p95=float(np.percentile(pred, 95)) if n >= 20 else float(np.max(pred)),

        trn_mean=float(np.mean(trn)),
        trn_upd_mean=float(np.mean(trn_upd)) if trn_upd.size else np.nan,
        trn_upd_p95=float(np.percentile(trn_upd, 95)) if trn_upd.size >= 20 else (float(np.max(trn_upd)) if trn_upd.size else np.nan),

        reb_mean=float(np.mean(reb)),
        wall_mean=float(np.mean(wall)),
        wall_p50=float(np.median(wall)),
        wall_p95=float(np.percentile(wall, 95)) if n >= 20 else float(np.max(wall)),

        wall_upd_mean=float(np.mean(wall_upd)) if wall_upd.size else np.nan,
        wall_upd_p95=float(np.percentile(wall_upd, 95)) if wall_upd.size >= 20 else (float(np.max(wall_upd)) if wall_upd.size else np.nan),
    )

# ------------------------------------------------------------
# Print a compact timing table
# ------------------------------------------------------------
print("\n=== Timing Summary (per-step) ===")
print("Units: ms (milliseconds) except update_freq")
for m in METHODS:
    if summ[m].get("n_steps", 0) <= 0:
        print(f"{m}: no steps logged")
        continue
    s = summ[m]
    print(
        f"{m}: steps={s['n_steps']}"
        f" | pred_mean={_ms(s['pred_mean']):.2f}ms (p95={_ms(s['pred_p95']):.2f}ms)"
        f" | wall_mean={_ms(s['wall_mean']):.2f}ms (p95={_ms(s['wall_p95']):.2f}ms)"
        f" | train_mean={_ms(s['trn_mean']):.2f}ms"
        f" | rebuild_mean={_ms(s['reb_mean']):.2f}ms"
        f" | update_freq={s['upd_freq']:.3f}"
        f" | train_on_update_mean={_ms(s['trn_upd_mean']):.2f}ms (p95={_ms(s['trn_upd_p95']):.2f}ms)"
    )

print("\n=== Timing Summary (update steps only) ===")
for m in METHODS:
    if summ[m].get("n_steps", 0) <= 0:
        continue
    s = summ[m]
    print(
        f"{m}: update_freq={s['upd_freq']:.3f}"
        f" | wall_on_update_mean={_ms(s['wall_upd_mean']):.2f}ms (p95={_ms(s['wall_upd_p95']):.2f}ms)"
        f" | train_on_update_mean={_ms(s['trn_upd_mean']):.2f}ms (p95={_ms(s['trn_upd_p95']):.2f}ms)"
    )

# ------------------------------------------------------------
# Plot 1: Running average wall time per step (concatenated)
# ------------------------------------------------------------
plt.figure(figsize=(10, 3.4))
for m in METHODS:
    if summ[m].get("n_steps", 0) <= 0:
        continue
    y = _running_avg(summ[m]["wall"])
    plt.plot(np.arange(y.size), y, linewidth=2.0, label=m)
plt.xlabel("timestep (concatenated)")
plt.ylabel("running avg wall/step (s)")
plt.title("Running average wall time per step")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Plot 2: Wall time distribution (boxplot)
# ------------------------------------------------------------
plt.figure(figsize=(9.5, 3.8))
data = []
labels = []
for m in METHODS:
    if summ[m].get("n_steps", 0) <= 0:
        continue
    data.append(summ[m]["wall"])
    labels.append(m)
if len(data):
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("wall time per step (s)")
    plt.title("Wall time per step distribution")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
else:
    plt.close()

# ------------------------------------------------------------
# Plot 3: Prediction time distribution (boxplot)
# ------------------------------------------------------------
plt.figure(figsize=(9.5, 3.8))
data = []
labels = []
for m in METHODS:
    if summ[m].get("n_steps", 0) <= 0:
        continue
    data.append(summ[m]["pred"])
    labels.append(m)
if len(data):
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("prediction/planning time per step (s)")
    plt.title("Prediction (planning) time per step distribution")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
else:
    plt.close()

# ------------------------------------------------------------
# Plot 4: Update-time distribution (train_time on update steps only)
# ------------------------------------------------------------
plt.figure(figsize=(9.5, 3.8))
data = []
labels = []
for m in METHODS:
    if summ[m].get("n_steps", 0) <= 0:
        continue
    upd_mask = summ[m]["upd"] >= 0.5
    trn_upd = summ[m]["trn"][upd_mask] if np.any(upd_mask) else np.array([], dtype=np.float64)
    if trn_upd.size:
        data.append(trn_upd)
        labels.append(m)
if len(data):
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("update train time (s) on update steps only")
    plt.title("Update-time distribution (train_time_step | update_flag==1)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
else:
    plt.close()

# ------------------------------------------------------------
# Plot 5: Mean breakdown bar chart (pred/train/rebuild) per step
# ------------------------------------------------------------
plt.figure(figsize=(9.5, 3.8))
x = np.arange(len(METHODS))
pred_means = [summ[m]["pred_mean"] if summ[m].get("n_steps", 0) > 0 else np.nan for m in METHODS]
trn_means  = [summ[m]["trn_mean"]  if summ[m].get("n_steps", 0) > 0 else np.nan for m in METHODS]
reb_means  = [summ[m]["reb_mean"]  if summ[m].get("n_steps", 0) > 0 else np.nan for m in METHODS]

# Stacked bars without specifying colors (default cycle)
plt.bar(x, pred_means, label="pred_mean")
plt.bar(x, trn_means, bottom=pred_means, label="train_mean")
plt.bar(x, reb_means, bottom=(np.asarray(pred_means) + np.asarray(trn_means)), label="rebuild_mean")

plt.xticks(x, METHODS, rotation=0)
plt.ylabel("mean time per step (s)")
plt.title("Mean time breakdown per step (pred + train + rebuild)")
plt.grid(True, axis="y", alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

print("\n✅ Cell 9 complete (timing-focused).")
