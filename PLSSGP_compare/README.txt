
Run this file:
    python run_all.py

Tune settings at the top of run_all.py (CONFIG dict).

Notes:
- engine/ contains the exact notebook-derived modular CartPole pipeline files.
- Each run is executed in a fresh Python process (worker_single_run.py) with N_RUNS=1 patched in config.py.
- aggregate_eval.py merges worker artifacts and saves timing plots as PNGs.
- To try another env, add a compatible env module to engine/ and set CONFIG['env_module'].


Env task abstraction (new):
- Shared task logic lives under engine/envs/ (reward/cost/success hooks).
- Current CartPole task: engine/envs/cartpole_swingup/task.py
- Preferred env module path: 'envs/cartpole_swingup/env.py' (adapter.py and carpole_env.py remain compatibility shims)
