import contextlib
import io
import pickle
import subprocess
import sys
from pathlib import Path

from suite_config import METHOD_ORDER_DEFAULT, SUITE_CONFIG



def _validated_suite_config() -> dict:
    cfg = dict(SUITE_CONFIG) if isinstance(SUITE_CONFIG, dict) else {}
    if not cfg:
        raise RuntimeError('suite_config.py must define a non-empty SUITE_CONFIG dict')
    return cfg

def _merge_run_artifact(run_pkl: Path, method_pkl: Path) -> None:
    merged = {}
    if run_pkl.exists():
        with open(run_pkl, 'rb') as f:
            prev = pickle.load(f)
        if isinstance(prev, dict):
            merged.update(prev)
    with open(method_pkl, 'rb') as f:
        cur = pickle.load(f)
    merged.setdefault('run_id', cur.get('run_id'))
    merged.setdefault('seed', cur.get('seed'))
    reg = dict(merged.get('EVAL_REGISTRY', {}) or {})
    reg.update(dict(cur.get('EVAL_REGISTRY', {}) or {}))
    merged['EVAL_REGISTRY'] = reg
    with open(run_pkl, 'wb') as f:
        pickle.dump(merged, f)


def _normalized_methods(cfg: dict):
    methods = cfg.get('methods', METHOD_ORDER_DEFAULT)
    if methods is None:
        methods = METHOD_ORDER_DEFAULT
    methods = [str(m).strip().upper() for m in methods if str(m).strip()]
    methods = [m for m in methods if m in METHOD_ORDER_DEFAULT]
    ordered = [m for m in METHOD_ORDER_DEFAULT if m in methods]
    # Preserve any extra future names after defaults (none today, but harmless)
    ordered += [m for m in methods if m not in ordered]
    return ordered


def _make_unified_dashboard(cfg: dict):
    mode = str(cfg.get('dashboard_mode', 'unified')).strip().lower()
    if mode != 'unified':
        return None
    try:
        from worker_method_run import PygameLiveViewer
        size = tuple(cfg.get('dashboard_window_size', (1600, 980)))
        caption = str(cfg.get('dashboard_caption', 'GP Comparison Dashboard'))
        viewer = PygameLiveViewer(enabled=True, every_steps=1, min_dt=0.02, size=size, caption=caption)
        return viewer
    except Exception as e:
        print(f'[suite] Unified dashboard unavailable: {e}')
        return None


def _stream_worker(cmd, *, cwd=None, env=None, dashboard=None):
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip('\n')
            _is_frame_evt = line.startswith('__LIVE_FRAME__ ')
            # avoid dumping huge base64 frame payloads to terminal in unified dashboard mode
            if not _is_frame_evt:
                print(line)
            if dashboard is not None:
                try:
                    dashboard.on_log_line(line)
                except Exception:
                    pass
        proc.stdout.close()
        return proc.wait()
    finally:
        if dashboard is not None:
            try:
                dashboard.draw(force=True)
            except Exception:
                pass


def main() -> None:
    here = Path(__file__).resolve().parent
    engine_dir = here / 'engine'
    worker = here / 'worker_method_run.py'
    agg = here / 'summarize_eval.py'
    cfg = _validated_suite_config()

    env_module = str(cfg.get('env_module', 'envs/cartpole_swingup/env.py'))
    n_runs = int(cfg.get('n_runs', 1))
    episodes_per_run = int(cfg.get('episodes_per_run', 1))
    max_steps_per_ep = int(cfg.get('max_steps_per_ep', 600))
    seed_base = int(cfg.get('seed_base', 0))
    live_render = bool(cfg.get('live_render', True))
    live_every_steps = int(cfg.get('live_every_steps', 50))
    progress_every_steps = int(cfg.get('progress_every_steps', 50))
    live_only_first_ep = bool(cfg.get('live_only_first_ep', False))
    live_only_first_run = bool(cfg.get('live_only_first_run', True))
    dashboard_mode = str(cfg.get('dashboard_mode', 'unified')).strip().lower()

    methods_to_run = _normalized_methods(cfg)
    if not methods_to_run:
        print('[suite] No methods configured in suite_config.py (SUITE_CONFIG["methods"]). Nothing to run.')
        return

    pyexe = str(cfg.get('python_executable', sys.executable))
    py_cmd = [pyexe]
    if bool(cfg.get('python_unbuffered', True)):
        py_cmd.append('-u')

    out_root = here / str(cfg.get('output_root', 'outputs'))
    runs_dir = out_root / str(cfg.get('runs_subdir', 'run_artifacts'))
    plots_dir = out_root / str(cfg.get('plots_subdir', 'plots'))
    tmp_root = out_root / str(cfg.get('worker_tmp_subdir', '_worker_tmp'))

    runs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    dashboard = _make_unified_dashboard(cfg)
    worker_viewer_mode = 'local'
    if dashboard_mode == 'unified':
        worker_viewer_mode = 'headless'
    elif dashboard_mode == 'off':
        worker_viewer_mode = 'none'
    elif dashboard_mode == 'per_worker':
        worker_viewer_mode = 'local'
    else:
        print(f'[suite] Unknown dashboard_mode={dashboard_mode!r}; falling back to unified if possible')
        worker_viewer_mode = 'headless' if dashboard is not None else 'local'

    print('=== gp_compare suite launcher (fresh-process workers) ===')
    print('engine:', engine_dir)
    print('env_module:', env_module)
    print('n_runs:', n_runs, 'episodes_per_run:', episodes_per_run, 'seed_base:', seed_base)
    print('methods to run:', methods_to_run)
    print('dashboard_mode:', dashboard_mode, '(worker_viewer_mode=', worker_viewer_mode, ')')
    print('outputs:', out_root)

    failures = []
    for run_id in range(int(n_runs)):
        seed = int(seed_base) + run_id
        out_pkl = runs_dir / f'run_{run_id:03d}.pkl'
        if out_pkl.exists():
            out_pkl.unlink()
        print(f"\n===== RUN {run_id + 1}/{n_runs} (seed={seed}) =====")
        for method_name in methods_to_run:
            method_pkl = tmp_root / f'run_{run_id:03d}__{method_name}.pkl'
            if method_pkl.exists():
                method_pkl.unlink()
            print(f"--- METHOD {method_name} (fresh process) ---")
            cmd = [
                *py_cmd, str(worker),
                '--engine-dir', str(engine_dir),
                '--run-id', str(run_id),
                '--seed', str(seed),
                '--out', str(method_pkl),
                '--only-method', str(method_name),
                '--env-module', env_module,
                '--episodes-per-run', str(episodes_per_run),
                '--max-steps-per-ep', str(max_steps_per_ep),
                '--live-render', '1' if live_render else '0',
                '--live-every-steps', str(live_every_steps),
                '--progress-every-steps', str(progress_every_steps),
                '--live-only-first-ep', '1' if live_only_first_ep else '0',
                '--live-only-first-run', '1' if live_only_first_run else '0',
                '--work-root', str(tmp_root),
                '--viewer-mode', worker_viewer_mode,
            ]
            rc = _stream_worker(cmd, cwd=str(here), dashboard=dashboard)
            if rc != 0:
                failures.append((run_id, method_name, rc))
                print(f'[suite] worker failed: run {run_id} method {method_name} rc={rc}')
                if bool(cfg.get('stop_on_first_worker_failure', True)):
                    break
            else:
                _merge_run_artifact(out_pkl, method_pkl)
        if failures and bool(cfg.get('stop_on_first_worker_failure', True)):
            break

    if failures:
        print('\nFailures:', failures)
        if bool(cfg.get('stop_on_first_worker_failure', True)):
            print('[suite] Skipping aggregate eval because a worker failed.')
            return

    print('\n=== Aggregating and saving PNG plots ===')
    cmd = [*py_cmd, str(agg), '--runs-dir', str(runs_dir), '--plots-dir', str(plots_dir)]
    rc = subprocess.call(cmd, cwd=str(here))
    if rc != 0:
        raise SystemExit(rc)

    print('\nDone.')
    print('Run artifacts:', runs_dir)
    print('Plot PNGs:', plots_dir)


if __name__ == '__main__':
    main()
