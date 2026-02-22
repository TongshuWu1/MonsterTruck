"""Suite-level configuration for the GP comparison launcher.

Edit this file for comparison runs (methods, runs/episodes, dashboard mode, output paths).
Keep model/environment/method hyperparameters in engine/config.py.
"""

import sys

# Canonical method display / execution order for the suite launcher.
METHOD_ORDER_DEFAULT = ['PALSGP', 'SVGP_GLOBAL', 'OSGPR_GLOBAL', 'EXACTGP_GLOBAL']

# Single source of truth for suite/runtime launcher settings.
SUITE_CONFIG = {
    # Process / paths
    'python_executable': sys.executable,
    'python_unbuffered': True,
    'output_root': 'outputs',
    'runs_subdir': 'run_artifacts',
    'plots_subdir': 'plots',
    'worker_tmp_subdir': '_worker_tmp',
    'stop_on_first_worker_failure': True,

    # What to run
    'env_module': 'envs/cartpole_swingup/env.py',
    'methods': ['PALSGP', 'SVGP_GLOBAL', 'OSGPR_GLOBAL'],  # add EXACTGP_GLOBAL if desired
    'n_runs': 1,
    'episodes_per_run': 2,
    'max_steps_per_ep': 600,
    'seed_base': 0,

    # Live progress / render cadence (forwarded into workers)
    'live_render': True,
    'live_every_steps': 50,
    'progress_every_steps': 50,
    'live_only_first_ep': False,
    'live_only_first_run': True,

    # Dashboard mode:
    #   'unified'    -> one pygame dashboard in run_suite.py; workers stream events to stdout
    #   'per_worker' -> each worker opens its own pygame window
    #   'off'        -> no dashboard / no live viewer windows
    'dashboard_mode': 'unified',

    # Unified dashboard window (used when dashboard_mode == 'unified')
    'dashboard_window_size': (1600, 980),
    'dashboard_caption': 'GP Comparison Dashboard',
}
