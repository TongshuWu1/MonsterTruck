import mujoco
from mujoco.glfw import glfw
import numpy as np
import os
import time

# ==============================
# Load model
# ==============================
xml_path = os.path.join(os.path.dirname(__file__), "monstertruck.model")
if not os.path.exists(xml_path):
    raise FileNotFoundError(f"XML file not found: {xml_path}")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# ==============================
# Initial spawn (Option 2 — flat, low upside-down start)
# ==============================
mujoco.mj_resetData(model, data)

# Slightly above the ground so the hood doesn't roll
data.qpos[:3] = np.array([0.0, 0.0, 0.2])   # tune 0.54–0.57 as needed
data.qvel[:] = 0.0
# Fully upside-down (180° about Y axis)
data.qpos[3:7] = np.array([0, 1, 0, 0])
mujoco.mj_forward(model, data)

# Check spawn height to verify no penetration
chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
print(f"Initial chassis COM z: {data.xpos[chassis_id,2]:.3f} m")

# ==============================
# Viewer setup
# ==============================
if not glfw.init():
    raise Exception("GLFW initialization failed")

window = glfw.create_window(1000, 800, "Monster Truck Follow Cam", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window creation failed")

glfw.make_context_current(window)
glfw.swap_interval(1)  # enable vsync (driver-dependent)

# Optional: report monitor refresh rate
try:
    mode = glfw.get_video_mode(glfw.get_primary_monitor())
    if mode:
        print(f"Monitor refresh: {mode.refresh_rate} Hz")
except Exception:
    pass

cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

cam.distance = 3.0
cam.elevation = -20
cam.azimuth = 90

# ==============================
# Keyboard Controls
# ==============================
throttle = 0.0
MAX_TORQUE = 1.0

def key_callback(window, key, scancode, action, mods):
    global throttle
    if action in [glfw.PRESS, glfw.REPEAT]:
        if key == glfw.KEY_W:
            throttle = MAX_TORQUE
        elif key == glfw.KEY_S:
            throttle = -MAX_TORQUE
        elif key == glfw.KEY_SPACE:
            throttle = 0.0
    elif action == glfw.RELEASE and key in [glfw.KEY_W, glfw.KEY_S]:
        throttle = 0.0

glfw.set_key_callback(window, key_callback)

# ==============================
# Simulation Loop with Follow Cam
# ==============================
print("Controls: W=Forward | S=Reverse | SPACE=Stop | ESC=Quit")
frame_skip = 10
REALTIME = True
timestep = model.opt.timestep
sim_start_wall = time.perf_counter()
last_print = sim_start_wall

while not glfw.window_should_close(window):
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        break

    # Apply throttle to both rear-wheel motors
    if model.nu >= 2:
        data.ctrl[0] = throttle
        data.ctrl[1] = throttle

    # Step physics faster than render
    for _ in range(frame_skip):
        mujoco.mj_step(model, data)

    # Real-time pacing
    if REALTIME:
        target_wall = sim_start_wall + data.time
        now = time.perf_counter()
        if now < target_wall:
            time.sleep(target_wall - now)

    # Update camera to follow the chassis
    chassis_pos = data.xpos[chassis_id].copy()
    cam.lookat[:] = chassis_pos
    cam.distance = 3.0
    cam.azimuth = 90
    cam.elevation = -25

    # Render
    w, h = glfw.get_framebuffer_size(window)
    if w > 0 and h > 0:
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL, scene)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

    # Print once per second
    now = time.perf_counter()
    if now - last_print > 1.0:
        speed_ratio = data.time / (now - sim_start_wall + 1e-9)
        print(f"Sim time: {data.time:.2f}s | speed ×{speed_ratio:.2f}")
        last_print = now

glfw.terminate()
print("Simulation terminated.")
