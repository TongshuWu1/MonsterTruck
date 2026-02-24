"""Suite-level configuration proxy.

Single source of truth now lives in engine/config.py (PROJECT_CONFIG).
Edit engine/config.py for suite + method + logging settings; keep environment dynamics
in the env module files.
"""

import ast
import sys
from pathlib import Path


def _load_from_engine_config():
    cfg_path = Path(__file__).resolve().parent / 'engine' / 'config.py'
    src = cfg_path.read_text(encoding='utf-8')
    tree = ast.parse(src, filename=str(cfg_path))
    vals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {'PROJECT_CONFIG', 'ENV_NAME', 'ENV_MODULE_BY_NAME'}:
                try:
                    vals[name] = ast.literal_eval(node.value)
                except Exception:
                    pass
    project = vals.get('PROJECT_CONFIG') if isinstance(vals.get('PROJECT_CONFIG'), dict) else {}
    meta = project.get('meta', {}) if isinstance(project.get('meta', {}), dict) else {}
    suite = dict(project.get('suite', {})) if isinstance(project.get('suite', {}), dict) else {}
    methods_cfg = project.get('methods', {}) if isinstance(project.get('methods', {}), dict) else {}
    enable = methods_cfg.get('enable', {}) if isinstance(methods_cfg.get('enable', {}), dict) else {}
    logging_cfg = project.get('logging', {}) if isinstance(project.get('logging', {}), dict) else {}

    method_order_default = list(meta.get('method_order_default', ['PALSGP', 'SVGP_GLOBAL', 'OSGPR_GLOBAL', 'EXACTGP_GLOBAL']))

    # Prefer method order + enabled toggles from PROJECT_CONFIG['methods']
    order_cfg = methods_cfg.get('order', method_order_default)
    ordered = [str(m).strip().upper() for m in order_cfg if str(m).strip()]
    valid = {str(m).strip().upper() for m in method_order_default}
    ordered = [m for m in ordered if m in valid]
    if ordered:
        suite['methods'] = [m for m in ordered if bool(enable.get(m, True))]

    # Prefer ENV_NAME mapping so env switch stays single-edit in engine/config.py
    env_name = vals.get('ENV_NAME')
    env_map = vals.get('ENV_MODULE_BY_NAME')
    if isinstance(env_name, str) and isinstance(env_map, dict):
        mapped = env_map.get(env_name)
        if isinstance(mapped, str) and mapped.strip():
            suite['env_module'] = mapped.strip()

    # launcher/runtime quality-of-life flags
    if 'suite_worker_stdout' in logging_cfg:
        suite['suite_worker_stdout'] = bool(logging_cfg.get('suite_worker_stdout'))
    if 'suite_progress_lines' in logging_cfg:
        suite['suite_progress_lines'] = bool(logging_cfg.get('suite_progress_lines'))

    suite.setdefault('python_executable', sys.executable)
    suite.setdefault('python_unbuffered', True)
    return method_order_default, suite


METHOD_ORDER_DEFAULT, SUITE_CONFIG = _load_from_engine_config()
