# GUI/GUI.py — Tk viewer for MuJoCo model (auto-discovers ../model/monstertruck.xml)
# deps: pip install mujoco glfw pillow

import os, sys
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import mujoco
from mujoco import mjtFontScale, mjtCatBit, MjvCamera, MjvOption, MjvScene, MjrContext, MjrRect
from mujoco.glfw import glfw


class MuJoCoTkViewer:
    def __init__(self, xml_path: Path, width=1000, height=800, fps=60):
        xml_path = Path(xml_path).expanduser().resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"XML not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data  = mujoco.MjData(self.model)

        # start upside-down & slightly raised (matches your env)
        self.data.qpos[:3]  = np.array([0.0, 0.0, 0.6])
        self.data.qpos[3:7] = np.array([0.0, 1.0, 0.0, 0.0])
        self.data.qvel[:]   = 0.0
        mujoco.mj_forward(self.model, self.data)

        # --- hidden GLFW context for offscreen rendering ---
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(width, height, "mujoco-offscreen", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")
        glfw.make_context_current(self.window)

        self.width, self.height = int(width), int(height)
        self.ms_per_frame = max(1, int(1000 / fps))

        # MuJoCo render state
        self.cam = MjvCamera()
        self.opt = MjvOption()
        self.scene = MjvScene(self.model, maxgeom=20000)
        self.ctx = MjrContext(self.model, mjtFontScale.mjFONTSCALE_150)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.0, 90, -25
        self.lookat_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

        # Tk UI
        self.root = tk.Tk()
        self.root.title(f"MonsterTruck — {xml_path.name}")
        self.label = tk.Label(self.root)
        self.label.pack(fill="both", expand=True)

        self.status = tk.StringVar(value="Keys: ←/→ azimuth, ↑/↓ elevation, +/- zoom, q quit")
        bar = tk.Frame(self.root); bar.pack(fill="x")
        tk.Label(bar, textvariable=self.status, anchor="w").pack(side="left", padx=8, pady=4)
        tk.Button(bar, text="Recenter", command=self._recenter).pack(side="right", padx=8, pady=4)

        # key bindings
        self.root.bind("<Left>",  lambda e: self._nudge_cam(az=-5))
        self.root.bind("<Right>", lambda e: self._nudge_cam(az=+5))
        self.root.bind("<Up>",    lambda e: self._nudge_cam(el=+3))
        self.root.bind("<Down>",  lambda e: self._nudge_cam(el=-3))
        self.root.bind("+",       lambda e: self._nudge_cam(dist=-0.1))
        self.root.bind("=",       lambda e: self._nudge_cam(dist=-0.1))  # '=' without shift
        self.root.bind("-",       lambda e: self._nudge_cam(dist=+0.1))
        self.root.bind("q",       lambda e: self.close())

        # draw & loop
        self._draw_once()
        self.root.after(self.ms_per_frame, self._tick)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _recenter(self):
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.0, 90, -25

    def _nudge_cam(self, az=0, el=0, dist=0):
        self.cam.azimuth   += az
        self.cam.elevation += el
        self.cam.distance   = max(0.2, self.cam.distance + dist)

    def _render_rgb(self):
        if self.lookat_body >= 0:
            self.cam.lookat[:] = self.data.xpos[self.lookat_body]

        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mjtCatBit.mjCAT_ALL, self.scene)
        vp = MjrRect(0, 0, self.width, self.height)
        mujoco.mjr_render(vp, self.scene, self.ctx)

        rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        depth = np.empty((self.height, self.width), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth, vp, self.ctx)
        return np.flipud(rgb)  # flip from bottom-up to top-down

    def _draw_once(self):
        frame = self._render_rgb()
        img = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(image=img)
        self.label.configure(image=self.photo)

        info = f"dist={self.cam.distance:.2f} az={self.cam.azimuth} el={self.cam.elevation}"
        self.status.set(f"{info} — ←/→, ↑/↓, +/- , q")
        glfw.poll_events()

    def _tick(self):
        self._draw_once()
        self.root.after(self.ms_per_frame, self._tick)

    def run(self):
        self.root.mainloop()

    def close(self):
        try: self.root.destroy()
        except Exception: pass
        try: self.ctx.free()
        except Exception: pass
        try:
            glfw.destroy_window(self.window)
            glfw.terminate()
        except Exception:
            pass


def find_xml_from_layout(cli_arg: str | None) -> Path:
    """Resolve model XML with these priorities:
       1) CLI arg, if provided
       2) ../model/monstertruck.xml relative to this file (your layout)
       3) ./model/monstertruck.xml relative to CWD
       4) ./monstertruck.xml relative to CWD
       5) ask user via file dialog
    """
    if cli_arg:
        p = Path(cli_arg).expanduser()
        if p.exists(): return p.resolve()

    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "model" / "monstertruck.xml",
        Path.cwd() / "model" / "monstertruck.xml",
        Path.cwd() / "monstertruck.xml",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    # fallback: ask the user
    root = tk.Tk(); root.withdraw()
    picked = filedialog.askopenfilename(
        title="Select a MuJoCo XML model",
        initialdir=(here.parent / "model"),
        filetypes=[("MuJoCo XML", "*.xml"), ("All files", "*.*")]
    )
    root.destroy()
    if not picked:
        messagebox.showerror("Model not found", "No XML selected. Exiting.")
        sys.exit(1)
    return Path(picked).resolve()


def main():
    xml = find_xml_from_layout(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"[GUI] Using model: {xml}")
    viewer = MuJoCoTkViewer(xml_path=xml, width=1000, height=800, fps=60)
    viewer.run()


if __name__ == "__main__":
    main()
