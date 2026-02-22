import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHOD_ORDER = ["PALSGP", "OSGPR_GLOBAL", "SVGP_GLOBAL", "EXACTGP_GLOBAL"]


def merge_eval_registries(payload_paths):
    merged = {}
    run_count = 0
    for p in payload_paths:
        with open(p, 'rb') as f:
            payload = pickle.load(f)
        reg = payload.get('EVAL_REGISTRY', {}) or {}
        run_count += 1
        for m, v in reg.items():
            if m not in merged:
                merged[m] = {'method': v.get('method', m), 'run_stats': [], 'run_traces': []}
            merged[m]['run_stats'].extend(v.get('run_stats', []))
            merged[m]['run_traces'].extend(v.get('run_traces', []))
    return merged, run_count


def _methods_in_order(reg):
    methods = [m for m in METHOD_ORDER if m in reg] + [m for m in reg.keys() if m not in METHOD_ORDER]
    return list(dict.fromkeys(methods))


def _as_1d(x, dtype=np.float64):
    if x is None:
        return np.array([], dtype=dtype)
    a = np.asarray(x, dtype=dtype).reshape(-1)
    return a


def _cat(reg, method, key):
    arrs = []
    for tr in reg[method].get('run_traces', []):
        if key in tr and tr[key] is not None:
            a = _as_1d(tr[key])
            if a.size:
                arrs.append(a)
    return np.concatenate(arrs) if arrs else np.array([], dtype=np.float64)


def _running_avg(x):
    x = _as_1d(x)
    return (np.cumsum(x) / (np.arange(x.size) + 1.0)) if x.size else x


def _pad_and_nanmean(arrs):
    if not arrs:
        return np.array([], dtype=np.float64)
    arrs = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrs if a is not None and len(a)]
    if not arrs:
        return np.array([], dtype=np.float64)
    L = max(len(a) for a in arrs)
    mat = np.full((len(arrs), L), np.nan, dtype=np.float64)
    for i, a in enumerate(arrs):
        mat[i, :len(a)] = a
    return np.nanmean(mat, axis=0)


def _pad_and_nanstd(arrs):
    if not arrs:
        return np.array([], dtype=np.float64)
    arrs = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrs if a is not None and len(a)]
    if not arrs:
        return np.array([], dtype=np.float64)
    L = max(len(a) for a in arrs)
    mat = np.full((len(arrs), L), np.nan, dtype=np.float64)
    for i, a in enumerate(arrs):
        mat[i, :len(a)] = a
    return np.nanstd(mat, axis=0)


def _summary_stats(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)), float(np.nanstd(x))


def summarize(reg):
    methods = _methods_in_order(reg)
    summ = {}
    for m in methods:
        pred = _cat(reg, m, 'pred_time_step')
        trn = _cat(reg, m, 'train_time_step')
        reb = _cat(reg, m, 'rebuild_time_step')
        wall = _cat(reg, m, 'wall_time_step')
        upd = _cat(reg, m, 'update_flag_step')
        lens = [a.size for a in [pred, trn, reb, wall] if a.size]
        if not lens:
            summ[m] = {'n_steps': 0}
            continue
        n = min(lens)
        pred = pred[:n] if pred.size else np.zeros((n,), dtype=np.float64)
        trn = trn[:n] if trn.size else np.zeros((n,), dtype=np.float64)
        reb = reb[:n] if reb.size else np.zeros((n,), dtype=np.float64)
        wall = wall[:n] if wall.size else (pred + trn + reb)
        upd = upd[:n] if upd.size else np.zeros((n,), dtype=np.float64)
        upd_mask = upd >= 0.5
        trn_upd = trn[upd_mask] if np.any(upd_mask) else np.array([], dtype=np.float64)
        wall_upd = wall[upd_mask] if np.any(upd_mask) else np.array([], dtype=np.float64)
        summ[m] = dict(
            n_steps=n, pred=pred, trn=trn, reb=reb, wall=wall, upd=upd,
            upd_freq=float(np.mean(upd_mask)) if n else np.nan,
            pred_mean=float(np.mean(pred)),
            pred_p95=float(np.percentile(pred, 95)) if n >= 20 else float(np.max(pred)),
            trn_mean=float(np.mean(trn)),
            reb_mean=float(np.mean(reb)),
            wall_mean=float(np.mean(wall)),
            wall_p95=float(np.percentile(wall, 95)) if n >= 20 else float(np.max(wall)),
            trn_upd_mean=float(np.mean(trn_upd)) if trn_upd.size else np.nan,
            wall_upd_mean=float(np.mean(wall_upd)) if wall_upd.size else np.nan,
            trn_upd_p95=float(np.percentile(trn_upd, 95)) if trn_upd.size >= 20 else (float(np.max(trn_upd)) if trn_upd.size else np.nan),
            wall_upd_p95=float(np.percentile(wall_upd, 95)) if wall_upd.size >= 20 else (float(np.max(wall_upd)) if wall_upd.size else np.nan),
        )
    return methods, summ


def _plot_overlay(curves, xlabel, ylabel, title, out_path: Path, figsize=(10, 3.2), ylog=False):
    plt.figure(figsize=figsize)
    any_plotted = False
    for name, (t, y) in curves.items():
        if t is None or y is None:
            continue
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        T = min(len(t), len(y))
        if T <= 1:
            continue
        good = np.isfinite(t[:T]) & np.isfinite(y[:T])
        if np.count_nonzero(good) <= 1:
            continue
        plt.plot(t[:T][good], y[:T][good], linewidth=2.0, label=name)
        any_plotted = True
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylog:
        plt.yscale('log')
    plt.grid(True, alpha=0.25)
    if any_plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _run_stats_arrays(reg, method):
    run_stats = reg.get(method, {}).get('run_stats', []) or []
    rewards = []
    succ = []
    t2 = []
    for rs in run_stats:
        rewards.append(float(rs.get('total_reward', np.nan)))
        succ.append(float(bool(rs.get('success', False))))
        fst = rs.get('first_success_t_global', None)
        t2.append(np.nan if fst is None else float(fst))
    return np.asarray(rewards, dtype=np.float64), np.asarray(succ, dtype=np.float64), np.asarray(t2, dtype=np.float64)


def _overlay_curve_from_runs(reg, method, kind):
    traces = reg.get(method, {}).get('run_traces', []) or []
    xs_runs, ys_runs = [], []
    if kind == 'pred_step_mean':
        for tr in traces:
            y = _as_1d(tr.get('pred_time_step'))
            if y.size:
                ys_runs.append(y)
                xs_runs.append(np.arange(y.size, dtype=np.float64))
    elif kind == 'wall_cum_excl_vis_mean':
        for tr in traces:
            wall = _as_1d(tr.get('wall_time_step'))
            if not wall.size:
                continue
            vis = _as_1d(tr.get('vis_time_step'))
            n = wall.size
            if vis.size:
                n = min(n, vis.size)
                wall_use = wall[:n] - vis[:n]
            else:
                wall_use = wall[:n]
            wall_use = np.maximum(wall_use, 0.0)
            y = np.cumsum(wall_use)
            ys_runs.append(y)
            xs_runs.append(np.arange(y.size, dtype=np.float64))
    elif kind in ('train_per_update_mean', 'train_cum_update_mean'):
        for tr in traces:
            trn = _as_1d(tr.get('train_time_step'))
            upd = _as_1d(tr.get('update_flag_step'))
            upd_mask = None
            if trn.size and upd.size:
                n = min(trn.size, upd.size)
                trn = trn[:n]
                upd_mask = upd[:n] >= 0.5
            elif trn.size:
                upd_mask = trn > 0
            else:
                continue
            y_upd = trn[upd_mask] if np.any(upd_mask) else np.array([], dtype=np.float64)
            if not y_upd.size:
                continue
            x_upd = _as_1d(tr.get('update_t_global'))
            if x_upd.size != y_upd.size:
                if upd.size and np.any(upd_mask):
                    # timestep index for update steps
                    x_upd = np.flatnonzero(upd_mask).astype(np.float64)
                else:
                    x_upd = np.arange(y_upd.size, dtype=np.float64)
            if kind == 'train_cum_update_mean':
                y_upd = np.cumsum(y_upd)
            xs_runs.append(x_upd)
            ys_runs.append(y_upd)
    else:
        raise ValueError(f'unknown overlay curve kind: {kind}')

    if not ys_runs:
        return None, None
    x_mean = _pad_and_nanmean(xs_runs)
    y_mean = _pad_and_nanmean(ys_runs)
    return x_mean, y_mean


def _save_requested_overlay_plots(reg, out_dir: Path):
    methods = _methods_in_order(reg)

    # 1) MPPI planning time per step (mean curve)
    curves_pred = {}
    for name in methods:
        curves_pred[name] = _overlay_curve_from_runs(reg, name, 'pred_step_mean')
    _plot_overlay(
        curves_pred,
        xlabel='timestep (concatenated episodes within run)',
        ylabel='MPPI planning time (s) per step',
        title='ALL METHODS — MPPI planning time per timestep (mean curves)',
        out_path=out_dir / '10_overlay_mppi_planning_time_mean_curves.png',
    )

    # 2) Wall-time cumulative (excluding visualization)
    curves_wall = {}
    for name in methods:
        curves_wall[name] = _overlay_curve_from_runs(reg, name, 'wall_cum_excl_vis_mean')
    _plot_overlay(
        curves_wall,
        xlabel='timestep (concatenated episodes within run)',
        ylabel='wall time cumulative (s) (excl vis)',
        title='ALL METHODS — Wall time cumulative (EXCLUDING visualization) (mean curves)',
        out_path=out_dir / '11_overlay_wall_time_cumulative_excl_vis_mean_curves.png',
    )

    # 3) Training time per UPDATE (update-only)
    curves_train_upd = {}
    for name in methods:
        curves_train_upd[name] = _overlay_curve_from_runs(reg, name, 'train_per_update_mean')
    _plot_overlay(
        curves_train_upd,
        xlabel='timestep (update steps only; x=mean update timestep by update index)',
        ylabel='training time per update (s)',
        title='ALL METHODS — Training time per UPDATE (update-only mean curves)',
        out_path=out_dir / '12_overlay_training_time_per_update_mean_curves.png',
    )

    # 4) Cumulative training time over UPDATEs (update-only)
    curves_train_cum = {}
    for name in methods:
        curves_train_cum[name] = _overlay_curve_from_runs(reg, name, 'train_cum_update_mean')
    _plot_overlay(
        curves_train_cum,
        xlabel='timestep (update steps only; x=mean update timestep by update index)',
        ylabel='cumulative training time (s)',
        title='ALL METHODS — Cumulative training time over UPDATEs (update-only mean curves)',
        out_path=out_dir / '13_overlay_cumulative_training_time_over_updates_mean_curves.png',
    )

    # 5) Run-level summary: reward / success / time-to-success
    method_names = methods
    rewards_map, succ_map, t2_map = {}, {}, {}
    for name in method_names:
        rr, ss, tt = _run_stats_arrays(reg, name)
        rewards_map[name] = rr
        succ_map[name] = ss
        t2_map[name] = tt

    # reward
    if all(rewards_map.get(n, np.array([])).size > 0 for n in method_names):
        reward_means, reward_stds = [], []
        for name in method_names:
            m, s = _summary_stats(rewards_map[name])
            reward_means.append(m); reward_stds.append(s)
        plt.figure(figsize=(10, 3.2))
        x = np.arange(len(method_names))
        plt.errorbar(x, reward_means, yerr=reward_stds, fmt='o', capsize=5)
        plt.xticks(x, method_names, rotation=25, ha='right')
        plt.ylabel('total reward (per run)')
        plt.title('ALL METHODS — Total reward across runs (mean ± std)')
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / '14_runlevel_total_reward_mean_std.png', dpi=160)
        plt.close()

    # success rate
    if all(succ_map.get(n, np.array([])).size > 0 for n in method_names):
        succ_means, succ_stds = [], []
        for name in method_names:
            m, s = _summary_stats(succ_map[name])
            succ_means.append(m); succ_stds.append(s)
        plt.figure(figsize=(10, 3.2))
        x = np.arange(len(method_names))
        plt.errorbar(x, succ_means, yerr=succ_stds, fmt='o', capsize=5)
        plt.xticks(x, method_names, rotation=25, ha='right')
        plt.ylim([-0.05, 1.05])
        plt.ylabel('success rate')
        plt.title('ALL METHODS — Success rate across runs (mean ± std)')
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / '15_runlevel_success_rate_mean_std.png', dpi=160)
        plt.close()

    # time-to-success (NaN if no success)
    if all(t2_map.get(n, np.array([])).size > 0 for n in method_names):
        t2_means, t2_stds = [], []
        for name in method_names:
            m, s = _summary_stats(t2_map[name])
            t2_means.append(m); t2_stds.append(s)
        plt.figure(figsize=(10, 3.2))
        x = np.arange(len(method_names))
        plt.errorbar(x, t2_means, yerr=t2_stds, fmt='o', capsize=5)
        plt.xticks(x, method_names, rotation=25, ha='right')
        plt.ylabel('time to success (timestep)')
        plt.title('ALL METHODS — Time-to-success across runs (mean ± std, NaN if no success)')
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / '16_runlevel_time_to_success_mean_std.png', dpi=160)
        plt.close()


def save_plots(reg, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    methods, summ = summarize(reg)
    print('Methods found:', methods)
    print('\n=== Timing Summary (per-step) ===')
    print('Units: ms (milliseconds) except update_freq')
    for m in methods:
        s = summ[m]
        if s.get('n_steps', 0) <= 0:
            print(f'{m}: no steps logged')
            continue
        print(
            f"{m}: steps={s['n_steps']}"
            f" | pred_mean={1e3*s['pred_mean']:.2f}ms (p95={1e3*s['pred_p95']:.2f}ms)"
            f" | wall_mean={1e3*s['wall_mean']:.2f}ms (p95={1e3*s['wall_p95']:.2f}ms)"
            f" | train_mean={1e3*s['trn_mean']:.2f}ms"
            f" | rebuild_mean={1e3*s['reb_mean']:.2f}ms"
            f" | update_freq={s['upd_freq']:.3f}"
            f" | train_on_update_mean={1e3*s['trn_upd_mean']:.2f}ms (p95={1e3*s['trn_upd_p95']:.2f}ms)"
        )

    # Existing timing-focused plots (kept)
    plt.figure(figsize=(10, 3.4))
    for m in methods:
        if summ[m].get('n_steps', 0) <= 0:
            continue
        y = _running_avg(summ[m]['wall'])
        plt.plot(np.arange(y.size), y, linewidth=2.0, label=m)
    plt.xlabel('timestep (concatenated)')
    plt.ylabel('running avg wall/step (s)')
    plt.title('Running average wall time per step')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / '01_running_average_wall_time_per_step.png', dpi=160)
    plt.close()

    plt.figure(figsize=(9.5, 3.8))
    data, labels = [], []
    for m in methods:
        if summ[m].get('n_steps', 0) <= 0:
            continue
        data.append(summ[m]['wall'])
        labels.append(m)
    if data:
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel('wall time per step (s)')
        plt.title('Wall time per step distribution')
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / '02_wall_time_distribution_boxplot.png', dpi=160)
    plt.close()

    plt.figure(figsize=(9.5, 3.8))
    data, labels = [], []
    for m in methods:
        if summ[m].get('n_steps', 0) <= 0:
            continue
        data.append(summ[m]['pred'])
        labels.append(m)
    if data:
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel('prediction/planning time per step (s)')
        plt.title('Prediction (planning) time per step distribution')
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / '03_prediction_time_distribution_boxplot.png', dpi=160)
    plt.close()

    plt.figure(figsize=(9.5, 3.8))
    data, labels = [], []
    for m in methods:
        if summ[m].get('n_steps', 0) <= 0:
            continue
        upd_mask = summ[m]['upd'] >= 0.5
        trn_upd = summ[m]['trn'][upd_mask] if np.any(upd_mask) else np.array([], dtype=np.float64)
        if trn_upd.size:
            data.append(trn_upd)
            labels.append(m)
    if data:
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel('update train time (s) on update steps only')
        plt.title('Update-time distribution (train_time_step | update_flag==1)')
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / '04_update_time_distribution_boxplot.png', dpi=160)
    plt.close()

    plt.figure(figsize=(9.5, 3.8))
    x = np.arange(len(methods))
    pred_means = [summ[m]['pred_mean'] if summ[m].get('n_steps', 0) > 0 else np.nan for m in methods]
    trn_means = [summ[m]['trn_mean'] if summ[m].get('n_steps', 0) > 0 else np.nan for m in methods]
    reb_means = [summ[m]['reb_mean'] if summ[m].get('n_steps', 0) > 0 else np.nan for m in methods]
    plt.bar(x, pred_means, label='pred_mean')
    plt.bar(x, trn_means, bottom=pred_means, label='train_mean')
    plt.bar(x, reb_means, bottom=(np.asarray(pred_means) + np.asarray(trn_means)), label='rebuild_mean')
    plt.xticks(x, methods, rotation=0)
    plt.ylabel('mean time per step (s)')
    plt.title('Mean time breakdown per step (pred + train + rebuild)')
    plt.grid(True, axis='y', alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / '05_mean_time_breakdown_stacked_bar.png', dpi=160)
    plt.close()

    # New overlay evaluation plots matching requested "Cell 9" style
    _save_requested_overlay_plots(reg, out_dir)

    with open(out_dir / 'merged_eval_registry.pkl', 'wb') as f:
        pickle.dump(reg, f)
    print(f"\nSaved plots to: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--plots-dir', required=True)
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    payloads = sorted(runs_dir.glob('run_*.pkl'))
    if not payloads:
        raise SystemExit(f'No run_*.pkl files found in {runs_dir}')
    reg, n = merge_eval_registries(payloads)
    print(f'Loaded {n} worker artifacts')
    save_plots(reg, Path(args.plots_dir))


if __name__ == '__main__':
    main()
