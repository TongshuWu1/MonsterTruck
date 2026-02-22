import argparse
import os
import sys
import runpy
import pickle
import shutil
import re
from pathlib import Path


def _parse_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def patch_config_text(text: str, *, seed_base: int, n_runs: int, episodes_per_run: int,
                      max_steps_per_ep: int, live_render: bool, live_every_steps: int,
                      progress_every_steps: int, live_only_first_ep: bool, live_only_first_run: bool) -> str:
    subs = {
        'SEED_BASE': int(seed_base),
        'N_RUNS': int(n_runs),
        'N_EPISODES_PER_RUN': int(episodes_per_run),
        'MAX_STEPS_PER_EP': int(max_steps_per_ep),
    }
    for k, v in subs.items():
        text = re.sub(rf'^{k}\s*=\s*.*$', f'{k} = {v}', text, flags=re.M)

    # Inject runtime knobs used via globals().get(...) in experiments.py
    injection = f"""
# worker-injected runtime knobs
LIVE_RENDER = {bool(live_render)}
LIVE_ONLY_FIRST_RUN = {bool(live_only_first_run)}
LIVE_ONLY_FIRST_EP = {bool(live_only_first_ep)}
LIVE_EVERY_STEPS = {int(live_every_steps)}
PROGRESS_EVERY_STEPS = {int(progress_every_steps)}
"""
    if '# worker-injected runtime knobs' in text:
        text = re.sub(r'\n# worker-injected runtime knobs\n(?:.*\n)*?(?=\n[^#]|\Z)', '\n' + injection.strip('\n') + '\n', text, flags=re.M)
    else:
        text = text.rstrip() + '\n\n' + injection.strip('\n') + '\n'
    return text


def patch_experiments_text(text: str, only_method: str = "") -> str:
    # Progress print cadence configurable (affects all method loops)
    text = text.replace(
        'ep_prog = TextProgress(MAX_STEPS_PER_EP, prefix=f"{METHOD} run{run} ep{ep} step ", every=10)',
        'ep_prog = TextProgress(MAX_STEPS_PER_EP, prefix=f"{METHOD} run{run} ep{ep} step ", every=int(globals().get("PROGRESS_EVERY_STEPS", 50)))'
    )

    # Print global update timing with pred_time too (lets dashboard plot decision-time sampled on updates)
    text = text.replace(
        '                trn_dt = float(time.perf_counter() - tu0)',
        '                trn_dt = float(time.perf_counter() - tu0)\n                print(f"{METHOD} global update | r{run} e{ep} t={t_global} | pred_time={pred_dt:.3f}s train_time={trn_dt:.3f}s")'
    )

    # Episode summary line (all method loops) for dashboard parsing, including wall/pred aggregates
    ep_marker = '        ep_hold_max_list.append(int(hold_max))'
    ep_repl = (
        '        ep_hold_max_list.append(int(hold_max))\n'
        '        _ep_sl = slice(max(0, len(wall_time_step) - int(ep_steps)), len(wall_time_step))\n'
        '        _ep_wall_arr = np.asarray(wall_time_step[_ep_sl], dtype=np.float64)\n'
        '        _ep_vis_arr  = np.asarray(vis_time_step[_ep_sl], dtype=np.float64) if len(vis_time_step) >= len(wall_time_step) else np.zeros_like(_ep_wall_arr)\n'
        '        _ep_pred_arr = np.asarray(pred_time_step[_ep_sl], dtype=np.float64)\n'
        '        _ep_wall_excl_vis = float(np.nansum(_ep_wall_arr) - np.nansum(_ep_vis_arr)) if _ep_wall_arr.size else 0.0\n'
        '        _cum_wall_excl_vis = float(np.nansum(np.asarray(wall_time_step, dtype=np.float64)) - np.nansum(np.asarray(vis_time_step, dtype=np.float64))) if len(wall_time_step) else 0.0\n'
        '        _ep_pred_mean = float(np.nanmean(_ep_pred_arr)) if _ep_pred_arr.size else float("nan")\n'
        '        print(f"[{METHOD}] episode | r{run} e{ep} | reward={float(ep_reward):.2f} steps={int(ep_steps)} hold_max={int(hold_max)} success={bool(ep_hold_complete_t is not None)} first_success_t={int(ep_hold_complete_t) if ep_hold_complete_t is not None else None} ep_wall_excl_vis={_ep_wall_excl_vis:.3f}s cum_wall_excl_vis={_cum_wall_excl_vis:.3f}s decision_mean={_ep_pred_mean:.6f}s")'
    )
    text = text.replace(ep_marker, ep_repl)

    # Add single-method execution support (fresh process per method)
    if '--only-method' not in text:
        text = text.replace(
            'ap.add_argument("--skip-eval", action="store_true")',
            'ap.add_argument("--skip-eval", action="store_true")\nap.add_argument("--only-method", default="")'
        )
    if '_ONLY_METHOD = (str(args.only_method).strip().upper() if str(args.only_method).strip() else "")' not in text:
        text = text.replace(
            'g = globals()\n_exec_file(here / args.env_module, g)',
            'g = globals()\n_ONLY_METHOD = (str(args.only_method).strip().upper() if str(args.only_method).strip() else "")\ndef _method_enabled(name: str) -> bool:\n    return (_ONLY_METHOD == "" or str(name).upper() == _ONLY_METHOD)\n_exec_file(here / args.env_module, g)'
        )

    def _wrap_method_block(src: str, method_name: str, next_marker: str) -> str:
        start_tok = f'METHOD = "{method_name}"'
        i0 = src.find(start_tok)
        if i0 < 0:
            return src
        i1 = src.find(next_marker, i0)
        if i1 < 0:
            return src
        block = src[i0:i1]
        if block.lstrip().startswith('if _method_enabled('):
            return src
        indented = ''.join(('    ' + ln if ln.strip() else ln) for ln in block.splitlines(True))
        wrapped = (
            f'if _method_enabled("{method_name}"):\n'
            + indented
            + 'else:\n'
            + f'    print("\\n⏭️ Skipping {method_name} (ONLY_METHOD=" + str(_ONLY_METHOD) + ")")\n'
        )
        return src[:i0] + wrapped + src[i1:]

    text = _wrap_method_block(text, 'PALSGP', '# ===========================\n# Cell 7')
    text = _wrap_method_block(text, 'SVGP_GLOBAL', '# ===========================\n# Cell 8')
    text = _wrap_method_block(text, 'OSGPR_GLOBAL', '# ===========================\n# Cell 8.5')

    text = text.replace(
        'if bool(globals().get("ENABLE_EXACTGP_GLOBAL", True)):',
        'if bool(globals().get("ENABLE_EXACTGP_GLOBAL", True)) and _method_enabled("EXACTGP_GLOBAL"):'
    )

    return text


def make_worker_copy(engine_dir: Path, work_dir: Path, only_method: str = "", **cfg) -> Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(engine_dir, work_dir)
    cfg_path = work_dir / 'config.py'
    cfg_path.write_text(
        patch_config_text(cfg_path.read_text(encoding='utf-8'), **cfg),
        encoding='utf-8'
    )
    exp_path = work_dir / 'experiments.py'
    exp_path.write_text(patch_experiments_text(exp_path.read_text(encoding='utf-8'), only_method=only_method), encoding='utf-8')
    return work_dir


class PygameLiveViewer:
    def __init__(self, enabled=True, every_steps=50, min_dt=0.03, size=(1600, 980), caption='PLSSGP Compare Live'):
        self.enabled = bool(enabled)
        self.every_steps = max(1, int(every_steps))
        self.min_dt = float(min_dt)
        self.size = tuple(size)
        self.caption = str(caption)
        self._last_step = -10**9
        self._last_t = 0.0
        self._pygame = None
        self._screen = None
        self._font = None
        self._font_small = None
        self._font_big = None
        self._frame_rgb = None
        self._frame_size = (900, 540)
        self.latest_overlay = ''
        self.current = {'method': None, 'run': None, 'ep': None, 'step': None, 'max_step': None, 'r': None, 'u': None, 'hold': None, 'hold_need': None, 'upd': None}
        self.methods = {}
        self.log_tail = []
        self._last_draw_force = 0.0

    def reset(self):
        self._last_step = -10**9
        self._last_t = 0.0
        self._frame_rgb = None
        self.latest_overlay = ''

    def _ensure(self):
        if not self.enabled:
            return False
        if self._pygame is None:
            try:
                import pygame
                pygame.init()
                pygame.font.init()
                self._pygame = pygame
                self._screen = pygame.display.set_mode(self.size)
                pygame.display.set_caption(self.caption)
                self._font = pygame.font.SysFont('consolas', 20) or pygame.font.SysFont(None, 20)
                self._font_small = pygame.font.SysFont('consolas', 16) or pygame.font.SysFont(None, 16)
                self._font_big = pygame.font.SysFont('consolas', 24, bold=True) or pygame.font.SysFont(None, 24)
            except Exception as e:
                print(f"⚠️ pygame live viewer unavailable: {e}")
                self.enabled = False
                return False
        return True

    def _pump(self):
        if not self._ensure():
            return False
        pg = self._pygame
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                self.enabled = False
                return False
        return True

    def _append_log(self, line):
        s = str(line)
        self.log_tail.append(s)
        if len(self.log_tail) > 14:
            self.log_tail = self.log_tail[-14:]

    def _method_state(self, name):
        if name not in self.methods:
            self.methods[name] = {
                'current_run': None, 'current_ep': None, 'current_step': None, 'max_step': None,
                'last_progress': None, 'last_update_time': None, 'last_update_t': None,
                'episode_rows': [], 'run_rows': [], 'update_rows': [],
                'progress_rows': [],
                'last_progress_elapsed_s': None,
                # live current-episode traces
                'trace_run': None,
                'trace_ep': None,
                'trace_t': [],
                'trace_u': [],
                'trace_r': [],
                'trace_hold': [],
                'trace_hold_need': None,
            }
        return self.methods[name]

    def _append_trace_point(self, meth, run, ep, t, r, u, hold, hold_need):
        ms = self._method_state(meth)

        # Reset trace when episode changes
        if (ms.get('trace_run') != int(run)) or (ms.get('trace_ep') != int(ep)):
            ms['trace_run'] = int(run)
            ms['trace_ep'] = int(ep)
            ms['trace_t'] = []
            ms['trace_u'] = []
            ms['trace_r'] = []
            ms['trace_hold'] = []
            ms['trace_hold_need'] = int(hold_need)

        # Avoid duplicate t points
        if ms['trace_t'] and int(t) == ms['trace_t'][-1]:
            ms['trace_u'][-1] = float(u)
            ms['trace_r'][-1] = float(r)
            ms['trace_hold'][-1] = int(hold)
        else:
            ms['trace_t'].append(int(t))
            ms['trace_u'].append(float(u))
            ms['trace_r'].append(float(r))
            ms['trace_hold'].append(int(hold))

        MAX_TRACE = 2000
        if len(ms['trace_t']) > MAX_TRACE:
            for k in ['trace_t', 'trace_u', 'trace_r', 'trace_hold']:
                ms[k] = ms[k][-MAX_TRACE:]

    def _parse_line(self, line):
        import re
        s = str(line).strip()
        if not s:
            return
        # Live overlay line from LIVE_VIEWER.maybe_update(...)
        m = re.match(
            r'^(?P<meth>[A-Z_]+)\s+\|\s+r(?P<run>\d+)\s+e(?P<ep>\d+)\s+t=(?P<t>\d+)\s+\|\s+'
            r'r=(?P<r>[+\-]?\d+\.\d+)\s+u=(?P<u>[+\-]?\d+\.\d+)\s+hold=(?P<hold>\d+)/(?:\s*)?(?P<holdn>\d+)$',
            s
        )
        if m:
            d = m.groupdict()
            meth = d['meth']
            run = int(d['run']); ep = int(d['ep']); t = int(d['t'])
            r = float(d['r']); u = float(d['u'])
            hold = int(d['hold']); hold_need = int(d['holdn'])

            ms = self._method_state(meth)
            ms['current_run'] = run
            ms['current_ep'] = ep
            ms['current_step'] = t
            ms['last_progress'] = dict(r=r, u=u, hold=hold, hold_need=hold_need, upd='?')

            self.current.update({
                'method': meth, 'run': run, 'ep': ep, 'step': t, 'max_step': ms.get('max_step'),
                'r': r, 'u': u, 'hold': hold, 'hold_need': hold_need, 'upd': self.current.get('upd')
            })

            self._append_trace_point(meth, run, ep, t, r, u, hold, hold_need)
            return

        # Step progress line (also captures elapsed seconds for live cumulative wall plot)
        m = re.match(
            r'^(?P<meth>[A-Z_]+)\s+run(?P<run>\d+)\s+ep(?P<ep>\d+)\s+step\s+'
            r'(?P<step>\d+)/(?:\s*)?(?P<max>\d+)'
            r'\s+\|\s+.*?\|\s+(?P<elapsed>\d+\.\d+)s\s+\|\s+'
            r'r=(?P<r>[+\-]\d+\.\d+)\s+u=(?P<u>[+\-]\d+\.\d+)\s+'
            r'hold=(?P<hold>\d+)/(?:\s*)?(?P<holdn>\d+)\s+upd=(?P<upd>[a-zA-Z])',
            s
        )
        if m:
            d = m.groupdict()
            meth = d['meth']
            run = int(d['run']); ep = int(d['ep'])
            step = int(d['step']); max_step = int(d['max'])
            elapsed_s = float(d['elapsed'])

            ms = self._method_state(meth)
            ms['current_run'] = run
            ms['current_ep'] = ep
            ms['current_step'] = step
            ms['max_step'] = max_step
            ms['last_progress_elapsed_s'] = elapsed_s
            ms['last_progress'] = dict(r=float(d['r']), u=float(d['u']), hold=int(d['hold']), hold_need=int(d['holdn']), upd=d['upd'])

            pr = ms['progress_rows']
            row = {'run': run, 'ep': ep, 'step': step, 'max_step': max_step, 'elapsed_s': elapsed_s}
            if pr and pr[-1]['run'] == run and pr[-1]['ep'] == ep and pr[-1]['step'] == step:
                pr[-1] = row
            else:
                pr.append(row)
            if len(pr) > 8000:
                ms['progress_rows'] = pr[-8000:]

            self.current.update({'method': meth, 'run': run, 'ep': ep, 'step': step, 'max_step': max_step,
                                 'r': float(d['r']), 'u': float(d['u']), 'hold': int(d['hold']), 'hold_need': int(d['holdn']), 'upd': d['upd']})

            # Handle occasional carriage-return concatenation, e.g. progress line + "METHOD global update ..."
            m_upd_suffix = re.search(r'([A-Z_]+ global update \| r\d+ e\d+ t=\d+ \| (?:(?:pred_time=\d+\.\d+s\s+)?)train_time=\d+\.\d+s)', s)
            if m_upd_suffix is not None:
                try:
                    self._parse_line(m_upd_suffix.group(1))
                except Exception:
                    pass
            return
        # Global update line
        m = re.search(r'(?P<meth>[A-Z_]+) global update \| r(?P<run>\d+) e(?P<ep>\d+) t=(?P<t>\d+) \| (?:(?:pred_time=(?P<pt>\d+\.\d+)s\s+)?)train_time=(?P<tt>\d+\.\d+)s', s)
        if m:
            d = m.groupdict(); meth = d['meth']; ms = self._method_state(meth)
            ms['last_update_time'] = float(d['tt'])
            ms['last_update_t'] = int(d['t'])
            ms['current_run'] = int(d['run'])
            ms['current_ep'] = int(d['ep'])
            ms['update_rows'].append({
                'run': int(d['run']), 'ep': int(d['ep']), 't': int(d['t']),
                'pred_time': (None if d.get('pt') in (None, '') else float(d['pt'])),
                'train_time': float(d['tt'])
            })
            if len(ms['update_rows']) > 4000:
                ms['update_rows'] = ms['update_rows'][-4000:]
            return
        # Episode summary line (worker injected)
        m = re.match(r'^\[(?P<meth>[A-Z_]+)\] episode \| r(?P<run>\d+) e(?P<ep>\d+) \| reward=(?P<rew>-?\d+\.\d+) steps=(?P<steps>\d+) hold_max=(?P<hold>\d+) success=(?P<succ>True|False) first_success_t=(?P<fst>None|\d+)(?: ep_wall_excl_vis=(?P<epwall>-?\d+\.\d+)s cum_wall_excl_vis=(?P<cumwall>-?\d+\.\d+)s decision_mean=(?P<dmean>(?:nan|[-+]?\d+\.\d+))s)?$', s)
        if m:
            d = m.groupdict(); meth = d['meth']; ms = self._method_state(meth)
            row = {
                'run': int(d['run']), 'ep': int(d['ep']), 'reward': float(d['rew']), 'steps': int(d['steps']),
                'hold_max': int(d['hold']), 'success': (d['succ'] == 'True'),
                'first_success_t': None if d['fst'] == 'None' else int(d['fst']),
                'ep_wall_excl_vis': (None if d.get('epwall') in (None, '') else float(d['epwall'])),
                'cum_wall_excl_vis': (None if d.get('cumwall') in (None, '') else float(d['cumwall'])),
                'decision_mean': (None if d.get('dmean') in (None, '') else float(d['dmean']) if str(d.get('dmean')).lower() != 'nan' else None),
            }
            ms['episode_rows'].append(row)
            if len(ms['episode_rows']) > 200:
                ms['episode_rows'] = ms['episode_rows'][-200:]
            return
        # Run summary line from experiments
        m = re.match(r'^\[(?P<meth>[A-Z_]+)\] run (?P<run1>\d+)/(?:\d+): reward=(?P<rew>-?\d+\.\d+) success=(?P<succ>True|False) first_success_t=(?P<fst>None|\d+)$', s)
        if m:
            d = m.groupdict(); meth = d['meth']; ms = self._method_state(meth)
            row = {
                'run_1based': int(d['run1']), 'reward': float(d['rew']), 'success': (d['succ'] == 'True'),
                'first_success_t': None if d['fst'] == 'None' else int(d['fst'])
            }
            ms['run_rows'].append(row)
            if len(ms['run_rows']) > 50:
                ms['run_rows'] = ms['run_rows'][-50:]
            return

    def on_log_line(self, line):
        self._append_log(line)
        self._parse_line(line)
        self.draw(force=False)

    def _draw_text(self, x, y, text, color=(230,230,230), font=None):
        pg = self._pygame
        if not text:
            return y
        if font is None:
            font = self._font
        surf = font.render(str(text), True, color)
        self._screen.blit(surf, (x, y))
        return y + surf.get_height() + 2

    def _blit_frame(self):
        if self._frame_rgb is None:
            return
        pg = self._pygame
        import numpy as np

        arr = np.asarray(self._frame_rgb)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return

        # Ensure stable dtype + contiguous memory before passing into pygame/SDL
        arr = np.asarray(arr[..., :3], dtype=np.uint8)
        arr_t = np.ascontiguousarray(arr.swapaxes(0, 1))  # pygame expects (W, H, 3)

        try:
            surf = pg.surfarray.make_surface(arr_t)
        except Exception:
            return

        frame_rect = pg.Rect(10, 10, 900, 540)
        surf = pg.transform.smoothscale(surf, (frame_rect.w, frame_rect.h))
        self._screen.blit(surf, frame_rect.topleft)
        pg.draw.rect(self._screen, (90,90,90), frame_rect, 1)
        if self.latest_overlay:
            overlay = pg.Surface((frame_rect.w, 28), flags=pg.SRCALPHA)
            overlay.fill((255, 255, 255, 220))
            self._screen.blit(overlay, frame_rect.topleft)
            if self._font is not None:
                txt = self._font.render(str(self.latest_overlay), True, (0, 0, 0))
                self._screen.blit(txt, (frame_rect.x + 8, frame_rect.y + 5))

    def _nice_limits(self, ys, symmetric=False, default=(-1.0, 1.0)):
        vals = [float(y) for y in ys if y is not None]
        if not vals:
            return default
        y0, y1 = min(vals), max(vals)
        if symmetric:
            m = max(abs(y0), abs(y1), 1e-6)
            y0, y1 = -m, m
        if abs(y1 - y0) < 1e-9:
            y0 -= 1.0
            y1 += 1.0
        pad = 0.08 * (y1 - y0)
        return (y0 - pad, y1 + pad)

    def _draw_series_panel(self, rect, title, xs, ys, color=(120, 220, 255), y_limits=None, zero_line=True, y_tick_fmt="{:.2f}"):
        pg = self._pygame
        pg.draw.rect(self._screen, (55, 55, 55), rect, 1)
        self._draw_text(rect.x + 8, rect.y + 6, title, color=(255,255,180), font=self._font)

        plot = pg.Rect(rect.x + 10, rect.y + 32, rect.w - 20, rect.h - 42)
        pg.draw.rect(self._screen, (25, 25, 25), plot)

        if not xs or not ys:
            self._draw_text(plot.x + 10, plot.y + 8, "No data yet", font=self._font_small)
            return

        xmin, xmax = int(xs[0]), int(xs[-1])
        if xmax <= xmin:
            xmax = xmin + 1

        if y_limits is None:
            y0, y1 = self._nice_limits(ys)
        else:
            y0, y1 = y_limits
        if abs(y1 - y0) < 1e-9:
            y0 -= 1.0
            y1 += 1.0

        for k in range(5):
            yv = y0 + (y1 - y0) * (k / 4.0)
            py = int(plot.bottom - (yv - y0) / (y1 - y0) * plot.h)
            pg.draw.line(self._screen, (45,45,45), (plot.x, py), (plot.right, py), 1)
            self._draw_text(plot.x + 4, py - 10, y_tick_fmt.format(yv), color=(170,170,170), font=self._font_small)

        for k in range(5):
            xv = xmin + (xmax - xmin) * (k / 4.0)
            px = int(plot.x + (xv - xmin) / (xmax - xmin) * plot.w)
            pg.draw.line(self._screen, (45,45,45), (px, plot.y), (px, plot.bottom), 1)
            self._draw_text(px - 10, plot.bottom - 18, f"{int(xv)}", color=(170,170,170), font=self._font_small)

        if zero_line and (y0 < 0.0 < y1):
            py0 = int(plot.bottom - (0.0 - y0) / (y1 - y0) * plot.h)
            pg.draw.line(self._screen, (90,90,90), (plot.x, py0), (plot.right, py0), 1)

        pts = []
        for x, y in zip(xs, ys):
            px = plot.x + int((float(x) - xmin) / (xmax - xmin) * max(1, plot.w - 1))
            py = plot.bottom - int((float(y) - y0) / (y1 - y0) * max(1, plot.h - 1))
            pts.append((px, py))

        if len(pts) >= 2:
            pg.draw.lines(self._screen, color, False, pts, 2)
        elif len(pts) == 1:
            pg.draw.circle(self._screen, color, pts[0], 2)

    def _draw_action_trace_plot(self):
        pg = self._pygame
        rect = pg.Rect(10, 560, 900, 190)
        cur_meth = self.current.get('method')
        if not cur_meth or cur_meth not in self.methods:
            self._draw_series_panel(rect, "Action u(t) (current episode)", [], [])
            return

        ms = self.methods[cur_meth]
        xs = ms.get('trace_t', [])
        ys = ms.get('trace_u', [])

        if ys:
            m = max(1.0, max(abs(float(v)) for v in ys[-500:]))
            y_limits = (-1.1*m, 1.1*m)
        else:
            y_limits = (-1.0, 1.0)

        title = f"Action u(t) | {cur_meth} r{ms.get('trace_run')} e{ms.get('trace_ep')}"
        self._draw_series_panel(rect, title, xs, ys, color=(120, 220, 255), y_limits=y_limits, zero_line=True, y_tick_fmt="{:.2f}")

    def _draw_reward_hold_plot(self):
        pg = self._pygame
        rect = pg.Rect(10, 760, 900, 190)
        pg.draw.rect(self._screen, (55,55,55), rect, 1)
        self._draw_text(rect.x + 8, rect.y + 6, "Reward r(t) + Hold progress (current episode)", color=(255,255,180), font=self._font)

        cur_meth = self.current.get('method')
        plot = pg.Rect(rect.x + 10, rect.y + 32, rect.w - 20, rect.h - 42)
        pg.draw.rect(self._screen, (25,25,25), plot)
        if not cur_meth or cur_meth not in self.methods:
            self._draw_text(plot.x + 10, plot.y + 8, "No data yet", font=self._font_small)
            return

        ms = self.methods[cur_meth]
        xs = ms.get('trace_t', [])
        rs = ms.get('trace_r', [])
        hs = ms.get('trace_hold', [])
        hold_need = ms.get('trace_hold_need') or 1

        if not xs:
            self._draw_text(plot.x + 10, plot.y + 8, "No data yet", font=self._font_small)
            return

        xmin, xmax = int(xs[0]), int(xs[-1])
        if xmax <= xmin:
            xmax = xmin + 1

        r0, r1 = self._nice_limits(rs, symmetric=False, default=(-1, 1))
        if abs(r1 - r0) < 1e-9:
            r0 -= 1; r1 += 1

        for k in range(5):
            py = int(plot.y + k * (plot.h - 1) / 4.0)
            pg.draw.line(self._screen, (45,45,45), (plot.x, py), (plot.right, py), 1)

        if hold_need > 0:
            py = plot.bottom - int((hold_need / hold_need) * (plot.h - 1))
            pg.draw.line(self._screen, (80, 180, 80), (plot.x, py), (plot.right, py), 1)

        r_pts = []
        for x, y in zip(xs, rs):
            px = plot.x + int((x - xmin) / (xmax - xmin) * max(1, plot.w - 1))
            py = plot.bottom - int((float(y) - r0) / (r1 - r0) * max(1, plot.h - 1))
            r_pts.append((px, py))
        if len(r_pts) >= 2:
            pg.draw.lines(self._screen, (255, 190, 80), False, r_pts, 2)

        h_pts = []
        for x, h in zip(xs, hs):
            px = plot.x + int((x - xmin) / (xmax - xmin) * max(1, plot.w - 1))
            h_clamped = max(0.0, min(float(hold_need), float(h)))
            py = plot.bottom - int((h_clamped / max(1.0, float(hold_need))) * max(1, plot.h - 1))
            h_pts.append((px, py))
        if len(h_pts) >= 2:
            pg.draw.lines(self._screen, (80, 220, 120), False, h_pts, 2)

        self._draw_text(plot.x + 8, plot.y + 4, "orange: reward", color=(255,190,80), font=self._font_small)
        self._draw_text(plot.x + 120, plot.y + 4, f"green: hold / {hold_need}", color=(80,220,120), font=self._font_small)

    def _draw_method_panels(self):
        pg = self._pygame
        x0 = 930
        y = 10
        y = self._draw_text(x0, y, 'PLSSGP Compare Dashboard', color=(255,255,180), font=self._font_big)
        cur = self.current
        if cur.get('method') is not None:
            y = self._draw_text(x0, y+4, f"Current: {cur['method']}  r{cur['run']} e{cur['ep']}  step {cur['step']}/{cur['max_step']}")
            y = self._draw_text(x0, y, f"reward={cur['r']:+.3f}  u={cur['u']:+.3f}  hold={cur['hold']}/{cur['hold_need']}  upd={cur['upd']}")
        else:
            y = self._draw_text(x0, y+4, 'Current: waiting for progress...')
        y += 6
        order = ['PALSGP', 'OSGPR_GLOBAL', 'SVGP_GLOBAL', 'EXACTGP_GLOBAL'] + [k for k in self.methods.keys() if k not in {'PALSGP','OSGPR_GLOBAL','SVGP_GLOBAL','EXACTGP_GLOBAL'}]
        seen = set()
        for meth in order:
            if meth in seen or meth not in self.methods:
                continue
            seen.add(meth)
            ms = self.methods[meth]
            pg.draw.rect(self._screen, (70,70,70), pg.Rect(x0-4, y-2, 650, 88), 1)
            y = self._draw_text(x0, y, meth, color=(180,255,255), font=self._font)
            line = f"r{ms['current_run']} e{ms['current_ep']} step {ms['current_step']}/{ms['max_step']}" if ms['current_step'] is not None else 'no progress yet'
            y = self._draw_text(x0, y, line, font=self._font_small)
            lp = ms.get('last_progress') or {}
            if lp:
                y = self._draw_text(x0, y, f"r={lp.get('r',0):+.3f}  u={lp.get('u',0):+.3f}  hold={lp.get('hold')}/{lp.get('hold_need')}  upd={lp.get('upd')}", font=self._font_small)
            if ms.get('last_update_time') is not None:
                y = self._draw_text(x0, y, f"last global update: t={ms['last_update_t']}  train_time={ms['last_update_time']:.3f}s", color=(255,220,180), font=self._font_small)
            if ms['episode_rows']:
                e = ms['episode_rows'][-1]
                y = self._draw_text(x0, y, f"last ep: r{e['run']} e{e['ep']} reward={e['reward']:.1f} steps={e['steps']} hold={e['hold_max']} success={e['success']}", color=(200,255,200), font=self._font_small)
            else:
                y = self._draw_text(x0, y, "last ep: --", font=self._font_small)
            y += 8
        # Left-bottom area is used for live traces (action/reward/hold).

    def _method_color(self, meth):
        return {
            'PALSGP': (80,220,120),
            'OSGPR_GLOBAL': (80,160,255),
            'SVGP_GLOBAL': (255,190,80),
            'EXACTGP_GLOBAL': (210,120,255),
        }.get(meth, (220,220,220))

    def _draw_multi_method_overlay_panel(self, rect, title, series, y_label=None, y_tick_fmt="{:.3f}"):
        pg = self._pygame
        pg.draw.rect(self._screen, (55,55,55), rect, 1)
        self._draw_text(rect.x+8, rect.y+6, title, color=(255,255,180), font=self._font)
        plot = pg.Rect(rect.x+10, rect.y+34, rect.w-20, rect.h-60)
        pg.draw.rect(self._screen, (25,25,25), plot)

        series = {k:v for k,v in series.items() if v and len(v[0]) and len(v[1])}
        if not series:
            self._draw_text(plot.x+10, plot.y+10, 'No data yet', font=self._font_small)
            return

        all_x = [float(x) for xs, ys in series.values() for x in xs]
        all_y = [float(y) for xs, ys in series.values() for y in ys if y is not None]
        if not all_x or not all_y:
            self._draw_text(plot.x+10, plot.y+10, 'No numeric data yet', font=self._font_small)
            return
        xmin, xmax = min(all_x), max(all_x)
        if xmax <= xmin:
            xmax = xmin + 1.0
        ymin, ymax = min(all_y), max(all_y)
        if abs(ymax - ymin) < 1e-12:
            ymin -= 1.0; ymax += 1.0
        pad = 0.08 * (ymax - ymin)
        ymin -= pad; ymax += pad

        for k in range(5):
            yv = ymin + (ymax - ymin) * k / 4.0
            py = int(plot.bottom - (yv - ymin) / (ymax - ymin) * plot.h)
            pg.draw.line(self._screen, (45,45,45), (plot.x, py), (plot.right, py), 1)
            self._draw_text(plot.x+4, py-10, y_tick_fmt.format(yv), color=(170,170,170), font=self._font_small)
        for k in range(5):
            xv = xmin + (xmax - xmin) * k / 4.0
            px = int(plot.x + (xv - xmin) / (xmax - xmin) * plot.w)
            pg.draw.line(self._screen, (45,45,45), (px, plot.y), (px, plot.bottom), 1)
            self._draw_text(px-8, plot.bottom-18, f"{int(round(xv))}", color=(170,170,170), font=self._font_small)

        for meth, (xs, ys) in series.items():
            col = self._method_color(meth)
            pts = []
            for x, y in zip(xs, ys):
                if y is None:
                    continue
                px = plot.x + int((float(x) - xmin) / (xmax - xmin) * max(1, plot.w - 1))
                py = plot.bottom - int((float(y) - ymin) / (ymax - ymin) * max(1, plot.h - 1))
                pts.append((px, py))
            if len(pts) >= 2:
                pg.draw.lines(self._screen, col, False, pts, 2)
            for p in pts[-80:]:
                pg.draw.circle(self._screen, col, p, 2)

        # legend
        lx = rect.x + 12
        ly = rect.bottom - 22
        order = ['PALSGP','OSGPR_GLOBAL','SVGP_GLOBAL','EXACTGP_GLOBAL'] + [k for k in series.keys() if k not in {'PALSGP','OSGPR_GLOBAL','SVGP_GLOBAL','EXACTGP_GLOBAL'}]
        seen = set()
        for meth in order:
            if meth in seen or meth not in series:
                continue
            seen.add(meth)
            col = self._method_color(meth)
            pg.draw.rect(self._screen, col, pg.Rect(lx, ly+4, 12, 12))
            self._draw_text(lx+18, ly, meth, font=self._font_small)
            lx += 160
            if lx > rect.right - 150:
                lx = rect.x + 12
                ly -= 18

    def _draw_live_eval_plot(self):
        # Right-side live overlays:
        # 1) cumulative wall time (live-updating during current episode from progress elapsed)
        # 2) decision time at update events = pred_time + train_time
        pg = self._pygame
        rect1 = pg.Rect(930, 360, 650, 285)
        rect2 = pg.Rect(930, 655, 650, 295)

        # 1) cumulative wall time (live during episode)
        # x-axis = episode index (fractional while current episode is running)
        series_wall = {}
        for meth, ms in self.methods.items():
            xs, ys = [], []
            completed_rows = ms.get('episode_rows', [])
            cum_completed = 0.0
            for i, er in enumerate(completed_rows, start=1):
                v = er.get('cum_wall_excl_vis')
                if v is None:
                    ev = er.get('ep_wall_excl_vis')
                    if ev is None:
                        continue
                    cum_completed += float(ev)
                    v = cum_completed
                else:
                    v = float(v)
                    cum_completed = v
                xs.append(float(i))
                ys.append(v)

            pr = ms.get('progress_rows', [])
            if pr:
                last = pr[-1]
                cur_step = int(last.get('step', 0))
                cur_max = max(1, int(last.get('max_step', 1)))
                cur_elapsed = float(last.get('elapsed_s', 0.0))
                n_completed = len(completed_rows)
                x_live = float(n_completed) + (cur_step / float(cur_max))
                y_live = float(cum_completed) + cur_elapsed
                xs.append(x_live)
                ys.append(y_live)

            if xs:
                series_wall[meth] = (xs, ys)

        self._draw_multi_method_overlay_panel(
            rect1,
            'Live overlay — cumulative wall time (live during episode; progress elapsed)',
            series_wall,
            y_tick_fmt='{:.1f}'
        )

        # 2) decision time at updates = pred_time + train_time
        series_dec_upd = {}
        for meth, ms in self.methods.items():
            xs, ys = [], []
            upd_rows = ms.get('update_rows', [])
            j = 0
            for ur in upd_rows:
                tt = ur.get('train_time')
                if tt is None:
                    continue
                pt = ur.get('pred_time')
                pt = 0.0 if pt is None else float(pt)
                total_dec = pt + float(tt)
                j += 1
                xs.append(j)
                ys.append(total_dec)
            if xs:
                series_dec_upd[meth] = (xs, ys)

        self._draw_multi_method_overlay_panel(
            rect2,
            'Live overlay — decision time at updates (pred_time + train_time)',
            series_dec_upd,
            y_tick_fmt='{:.4f}'
        )

    def draw(self, force=False):
        import time
        if not self.enabled:
            return
        now = time.perf_counter()
        if (not force) and ((now - self._last_draw_force) < self.min_dt):
            return
        if not self._pump():
            return
        pg = self._pygame
        self._screen.fill((18, 18, 18))
        self._blit_frame()
        self._draw_method_panels()
        self._draw_live_eval_plot()
        self._draw_action_trace_plot()
        self._draw_reward_hold_plot()
        pg.display.flip()
        self._last_draw_force = now

    def maybe_update(self, step_i, x, theta, text=''):
        if not self.enabled:
            return
        import time
        now = time.perf_counter()
        if (int(step_i) - self._last_step) < self.every_steps and (now - self._last_t) < self.min_dt:
            return
        if not self._ensure():
            return
        if not self._pump():
            return
        fn = getattr(sys.modules.get('__main__'), 'render_cartpole_frame_from_state', None)
        if fn is None:
            return
        try:
            frame = fn(float(x), float(theta), x_threshold=2.4, W=self._frame_size[0], H=self._frame_size[1])
        except TypeError:
            frame = fn(float(x), float(theta), W=self._frame_size[0], H=self._frame_size[1])
        self._frame_rgb = frame
        self.latest_overlay = str(text) if text else ''
        if text:
            self._append_log(text)
            self._parse_line(text)
        self.draw(force=True)
        self._last_step = int(step_i)
        self._last_t = now


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine-dir', required=True)
    ap.add_argument('--run-id', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--env-module', default='envs/cartpole_swingup/env.py')
    ap.add_argument('--only-method', default='')
    ap.add_argument('--episodes-per-run', type=int, default=3)
    ap.add_argument('--max-steps-per-ep', type=int, default=600)
    ap.add_argument('--live-render', default='1')
    ap.add_argument('--live-every-steps', type=int, default=50)
    ap.add_argument('--progress-every-steps', type=int, default=50)
    ap.add_argument('--live-only-first-ep', default='0')
    ap.add_argument('--live-only-first-run', default='1')
    ap.add_argument('--work-root', default=None)
    args = ap.parse_args()

    engine_dir = Path(args.engine_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root).resolve() if args.work_root else (out_path.parent / '_worker_tmp')
    _method_tag = (str(args.only_method).strip() or 'ALL').replace('/', '_')
    work_dir = work_root / f'run_{args.run_id:03d}__{_method_tag}'

    make_worker_copy(
        engine_dir, work_dir,
        seed_base=args.seed,
        n_runs=1,
        episodes_per_run=args.episodes_per_run,
        max_steps_per_ep=args.max_steps_per_ep,
        live_render=_parse_bool(args.live_render),
        live_every_steps=args.live_every_steps,
        progress_every_steps=args.progress_every_steps,
        live_only_first_ep=_parse_bool(args.live_only_first_ep),
        live_only_first_run=_parse_bool(args.live_only_first_run),
        only_method=str(args.only_method or ''),
    )

    cwd_prev = os.getcwd()
    sys_path_prev = list(sys.path)
    argv_prev = list(sys.argv)
    try:
        os.chdir(work_dir)
        if str(work_dir) not in sys.path:
            sys.path.insert(0, str(work_dir))
        sys.argv = ['experiments.py', '--skip-eval', '--env-module', args.env_module] + (['--only-method', str(args.only_method)] if str(args.only_method).strip() else [])
        init_globals = {
            'LIVE_RENDER': _parse_bool(args.live_render),
            'LIVE_EVERY_STEPS': int(args.live_every_steps),
            'PROGRESS_EVERY_STEPS': int(args.progress_every_steps),
            'LIVE_ONLY_FIRST_EP': _parse_bool(args.live_only_first_ep),
            'LIVE_ONLY_FIRST_RUN': _parse_bool(args.live_only_first_run),
            'LIVE_VIEWER': PygameLiveViewer(
                enabled=_parse_bool(args.live_render),
                every_steps=int(args.live_every_steps),
                size=(1600, 980),
                caption=f"PLSSGP Compare | run {args.run_id} | {str(args.only_method).strip() or 'ALL'}"
            )
        }
        import builtins
        _orig_print = builtins.print
        _viewer = init_globals.get('LIVE_VIEWER')
        def _hooked_print(*args, **kwargs):
            sep = kwargs.get('sep', ' ')
            end = kwargs.get('end', '\n')
            try:
                line = sep.join(str(a) for a in args)
            except Exception:
                line = ' '.join([str(a) for a in args])
            if _viewer is not None and hasattr(_viewer, 'on_log_line'):
                try:
                    _viewer.on_log_line(line)
                except Exception:
                    pass
            return _orig_print(*args, **kwargs)
        builtins.print = _hooked_print
        try:
            g = runpy.run_path(str(work_dir / 'experiments.py'), init_globals=init_globals, run_name='__main__')
        finally:
            builtins.print = _orig_print
        payload = {
            'run_id': int(args.run_id),
            'seed': int(args.seed),
            'only_method': str(args.only_method or ''),
            'EVAL_REGISTRY': g.get('EVAL_REGISTRY', {}),
        }
        with open(out_path, 'wb') as f:
            pickle.dump(payload, f)
        print(f"\n[worker] saved run artifact -> {out_path}")
    finally:
        sys.argv = argv_prev
        sys.path[:] = sys_path_prev
        os.chdir(cwd_prev)


if __name__ == '__main__':
    main()
