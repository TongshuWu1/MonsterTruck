# Backward-compatibility shim (deprecated path)
# Canonical env implementation now lives at: engine/envs/cartpole_swingup/env.py
# Keeping this file avoids breaking older configs / notebook-style imports.
from envs.cartpole_swingup.env import *  # noqa: F401,F403
