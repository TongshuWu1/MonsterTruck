from pathlib import Path

def load_env_module(globals_dict, module_path="envs/cartpole_swingup/env.py"):
    code = Path(module_path).read_text(encoding="utf-8")
    exec(compile(code, module_path, "exec"), globals_dict, globals_dict)



def load_task_env_into_globals(globals_dict, env_name="cartpole_swingup"):
    """Future helper: populate task hooks and env symbols from a registered task env."""
    if str(env_name).lower() in ("cartpole_swingup", "carpole_swingup", "cartpole", "carpole"):
        # Backward-compatible: current implementation reuses the existing module.
        return load_env_module(globals_dict, module_path="envs/cartpole_swingup/env.py")
    if str(env_name).lower() in ("mountaincar", "mountaincar_hold", "continuous_mountaincar", "mountain_car"):
        return load_env_module(globals_dict, module_path="envs/mountaincar_hold/env.py")
    raise ValueError(f"No registered engine env adapter yet for env_name={env_name!r}")
