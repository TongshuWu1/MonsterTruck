Run the comparison suite:
    python run_suite.py

Primary files (clean/generic names):
- run_suite.py           -> suite launcher (fresh process per method); primary place to tune suite settings
- run_suite.py          -> edit SUITE_CONFIG at the top (main suite tuning location)
- suite_config.py        -> optional override file (kept for cleanliness / alternate presets)
- worker_method_run.py   -> executes one method in one fresh worker process
- summarize_eval.py      -> merges worker artifacts and writes PNG summary plots

Backward-compatible aliases (still work):
- run_suite.py             -> wrapper to run_suite.py
- worker_single_run.py   -> wrapper to worker_method_run.py
- aggregate_eval.py      -> wrapper to summarize_eval.py

Notes:
- engine/ contains the notebook-derived modular experiment pipeline.
- Worker patches are applied on a copied engine/ directory under outputs/_worker_tmp/ (per-worker sandbox).
- Unified dashboard mode is configured in suite_config.py via SUITE_CONFIG['dashboard_mode'].
  - 'unified' (recommended): one pygame dashboard for all methods
  - 'per_worker': one dashboard window per worker
  - 'off': no dashboard
- To try another env, add/use a compatible env module under engine/envs/ and set SUITE_CONFIG['env_module'].

Env task abstraction:
- Shared task logic lives under engine/envs/ (reward/cost/success hooks).
- Current CartPole task: engine/envs/cartpole_swingup/task.py
- Preferred env module path: 'envs/cartpole_swingup/env.py' (adapter.py and carpole_env.py remain compatibility shims)


Notes:
- run_all.py was removed to avoid duplicate entrypoints. Use run_suite.py only.
