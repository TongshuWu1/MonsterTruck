import mujoco
from mujoco.glfw import glfw
import numpy as np
import os
import time

# ==============================
# Load model
# ==============================
xml_path = os.path.join(os.path.dirname(__file__), "monstertruck.xml")
if not os.path.exists(xml_path):
    raise FileNotFoundError(f"XML file not found: {xml_path}")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

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
glfw.swap_interval(1)  # request VSync (driver may override on Windows)

# Report monitor refresh rate (helps explain speed differences)
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
    elif action == glfw.RELEASE:
        if key in [glfw.KEY_W, glfw.KEY_S]:
            throttle = 0.0

glfw.set_key_callback(window, key_callback)

# ==============================
# Simulation Loop with Follow Cam
# ==============================
print("Controls: W=Forward | S=Reverse | SPACE=Stop | ESC=Quit")
frame_skip = 10

# Real-time pacing toggle
REALTIME = True
timestep = model.opt.timestep  # 0.001 from your XML
sim_start_wall = time.perf_counter()  # wall-clock when sim begins

last_print = sim_start_wall

# Get chassis body id
chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

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

    # ---- Real-time pacing: sleep so that sim time ~= wall time ----
    if REALTIME:
        target_wall = sim_start_wall + data.time  # when we *should* be at in wall time
        now = time.perf_counter()
        if now < target_wall:
            time.sleep(target_wall - now)

    # Get chassis position & update camera to follow the car
    chassis_pos = data.xpos[chassis_id].copy()
    cam.lookat[:] = chassis_pos
    cam.distance = 3.0
    cam.azimuth = 90
    cam.elevation = -25

    # Render (guard against minimized window)
    w, h = glfw.get_framebuffer_size(window)
    if w > 0 and h > 0:
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

    # Optional: print simulation time once per second
    now = time.perf_counter()
    if now - last_print > 1.0:
        # Effective speed ratio: how many sim seconds per wall second
        # (since we're pacing, this should be ~1.0)
        speed_ratio = (data.time) / (now - sim_start_wall + 1e-9)
        print(f"Sim time: {data.time:.2f} s | speed x{speed_ratio:.2f}")
        last_print = now

glfw.terminate()
print("Simulation terminated.")
