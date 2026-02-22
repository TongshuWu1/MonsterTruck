import subprocess
import sys
import runpy
import io
import contextlib
from pathlib import Path

# =====================
# TUNE LAUNCHER/OUTPUT ONLY HERE
#   (experiment/suite knobs live in engine/config.py)
# =====================
CONFIG = {
    'python_executable': sys.executable,
    'output_root': 'outputs',
    'runs_subdir': 'run_artifacts',
    'plots_subdir': 'plots',
    'worker_tmp_subdir': '_worker_tmp',
    'stop_on_first_worker_failure': True,
}

METHOD_ORDER_DEFAULT = ['PALSGP', 'SVGP_GLOBAL', 'OSGPR_GLOBAL', 'EXACTGP_GLOBAL']

def _merge_run_artifact(run_pkl: Path, method_pkl: Path) -> None:
    import pickle
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




def _load_engine_runtime_config(engine_dir: Path) -> dict:
    cfg_path = engine_dir / 'config.py'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        g = runpy.run_path(str(cfg_path))
    # keep only plain scalar knobs we care about; fallbacks handled in main
    return g



def _methods_from_runtime_cfg(runtime_cfg: dict):
    enabled_map = {
        'PALSGP': bool(runtime_cfg.get('ENABLE_PALSGP', True)),
        'SVGP_GLOBAL': bool(runtime_cfg.get('ENABLE_SVGP_GLOBAL', True)),
        'OSGPR_GLOBAL': bool(runtime_cfg.get('ENABLE_OSGPR_GLOBAL', True)),
        'EXACTGP_GLOBAL': bool(runtime_cfg.get('ENABLE_EXACTGP_GLOBAL', True)),
    }

    method_order = list(runtime_cfg.get('METHOD_ORDER', METHOD_ORDER_DEFAULT))
    method_order = [m for m in method_order if m in METHOD_ORDER_DEFAULT]
    for m in METHOD_ORDER_DEFAULT:
        if m not in method_order:
            method_order.append(m)

    methods = [m for m in method_order if enabled_map.get(m, False)]
    return methods, enabled_map
def main() -> None:
    here = Path(__file__).resolve().parent
    engine_dir = here / 'engine'
    worker = here / 'worker_single_run.py'
    agg = here / 'aggregate_eval.py'
    runtime_cfg = _load_engine_runtime_config(engine_dir)

    env_module = str(runtime_cfg.get('ENV_MODULE', 'envs/cartpole_swingup/env.py'))
    n_runs = int(runtime_cfg.get('N_RUNS', 1))
    episodes_per_run = int(runtime_cfg.get('N_EPISODES_PER_RUN', 1))
    max_steps_per_ep = int(runtime_cfg.get('MAX_STEPS_PER_EP', 600))
    seed_base = int(runtime_cfg.get('SEED_BASE', 0))
    live_render = bool(runtime_cfg.get('LIVE_RENDER', True))
    live_every_steps = int(runtime_cfg.get('LIVE_EVERY_STEPS', 50))
    progress_every_steps = int(runtime_cfg.get('PROGRESS_EVERY_STEPS', 50))
    live_only_first_ep = bool(runtime_cfg.get('LIVE_ONLY_FIRST_EP', False))
    live_only_first_run = bool(runtime_cfg.get('LIVE_ONLY_FIRST_RUN', True))

    out_root = here / CONFIG['output_root']
    runs_dir = out_root / CONFIG['runs_subdir']
    plots_dir = out_root / CONFIG['plots_subdir']
    tmp_root = out_root / CONFIG['worker_tmp_subdir']

    runs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    methods_to_run, enabled_map = _methods_from_runtime_cfg(runtime_cfg)
    if not methods_to_run:
        print('[launcher] No methods are enabled in engine/config.py. Nothing to run.')
        return

    print('=== PLSSGP_compare fresh-process launcher ===')
    print('engine:', engine_dir)
    print('env_module:', env_module)
    print('n_runs:', n_runs, 'episodes_per_run:', episodes_per_run, 'seed_base:', seed_base)
    print('methods enabled map:', enabled_map)
    print('methods to run:', methods_to_run)
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
                CONFIG['python_executable'], str(worker),
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
            ]
            rc = subprocess.call(cmd)
            if rc != 0:
                failures.append((run_id, method_name, rc))
                print(f'[launcher] worker failed: run {run_id} method {method_name} rc={rc}')
                if CONFIG['stop_on_first_worker_failure']:
                    break
            else:
                _merge_run_artifact(out_pkl, method_pkl)
        if failures and CONFIG['stop_on_first_worker_failure']:
            break

    if failures:
        print('\nFailures:', failures)
        if CONFIG['stop_on_first_worker_failure']:
            print('Skipping aggregate eval because a worker failed.')
            return

    print('\n=== Aggregating and saving PNG plots ===')
    cmd = [CONFIG['python_executable'], str(agg), '--runs-dir', str(runs_dir), '--plots-dir', str(plots_dir)]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)

    print('\nDone.')
    print('Run artifacts:', runs_dir)
    print('Plot PNGs:', plots_dir)


if __name__ == '__main__':
    main()
