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

# Resolve wheel actuator IDs (by name); fallback to all actuators if not found
wheel_motor_names = ["front_left_motor", "front_right_motor", "rear_left_motor", "rear_right_motor"]
drive_ids, drive_names = [], []
for name in wheel_motor_names:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid != -1:
        drive_ids.append(aid)
        drive_names.append(name)
if not drive_ids:
    drive_ids = list(range(model.nu))
    drive_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}" for i in drive_ids]
print(f"Driving actuators: {drive_names} (ids: {drive_ids})")

# ==============================
# Initial spawn (upside-down)
# ==============================
mujoco.mj_resetData(model, data)
data.qpos[:3] = np.array([0.0, 0.0, 0.25])  # small lift off ground
data.qvel[:] = 0.0
data.qpos[3:7] = np.array([0, 1, 0, 0])    # 180° about X → upside-down
mujoco.mj_forward(model, data)

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
glfw.swap_interval(1)  # vsync

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
MAX_TORQUE = 1.0  # control ∈ [-1,1]; scaled by gear in XML

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
sim_start_wall = time.perf_counter()
last_print = sim_start_wall

while not glfw.window_should_close(window):
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        break

    # Apply throttle to ALL selected wheel motors
    if model.nu > 0:
        # zero all controls first (prevents stale values on other actuators)
        data.ctrl[:model.nu] = 0.0
        for aid in drive_ids:
            data.ctrl[aid] = throttle

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
        # Show current control values for debug
        ctrl_vals = np.round(data.ctrl[:model.nu], 3)
        print(f"Sim time: {data.time:.2f}s | speed ×{speed_ratio:.2f} | throttle {throttle:+.2f} | ctrl={ctrl_vals}")
        last_print = now

glfw.terminate()
print("Simulation terminated.")
