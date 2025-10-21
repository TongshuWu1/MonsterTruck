# evaluate_monstertruck.py
# Clean evaluation for MonsterTruckFlipEnv
# Renders every 20th episode in real time and summarizes success statistics

import os, math, time
import numpy as np
import matplotlib.pyplot as plt
from QLearning_Monstertruck import MonsterTruckFlipEnv, QAgent
from collections import Counter


def evaluate(episodes=100, max_steps=2000, save_dir="MonsterTruck_eval/Qtable_eval"):
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------- Load Agent ----------------------
    agent = QAgent()
    if not agent.load():
        print("⚠️ No trained Q-table found. Please train first.")
        return
    agent.eps = 0.0  # greedy evaluation

    total_success = 0
    all_rewards, all_steps, all_success_flags = [], [], []
    all_angle_timelines, all_angspeed_timelines = [], []

    # ---------------------- Evaluation Loop ----------------------
    for ep in range(1, episodes + 1):
        render = (ep % 20 == 0)
        realtime = render
        env = MonsterTruckFlipEnv(render=render, max_steps=max_steps, realtime=realtime)
        agent.actions = env.actions  # sync with env

        obs = env.reset()
        ep_ret, steps = 0.0, 0
        success_flag = False
        angle_timeline, angspeed_timeline = [], []

        for t in range(max_steps):
            a = agent.act(obs)
            obs, r, done, info = env.step(a)
            ep_ret += r
            steps += 1

            R = env.data.xmat[env.body_id].reshape(3, 3)
            angle_timeline.append(math.degrees(math.acos(np.clip(R[2, 2], -1, 1))))
            angspeed_timeline.append(float(np.linalg.norm(env.data.cvel[env.body_id][3:])))

            if done:
                success_flag = bool(info.get("success", False))
                break

        env.close()
        total_success += int(success_flag)
        all_success_flags.append(int(success_flag))
        all_rewards.append(ep_ret)
        all_steps.append(steps)

        # pad for averaging
        def pad_to(x, n): return np.pad(x, (0, max(0, n - len(x))), constant_values=np.nan)
        all_angle_timelines.append(pad_to(angle_timeline, max_steps))
        all_angspeed_timelines.append(pad_to(angspeed_timeline, max_steps))

        if ep % 10 == 0:
            print(f"Ep {ep:3d}/{episodes} | Ret: {ep_ret:7.2f} | Steps: {steps:4d} | Success: {success_flag}")

    # ---------------------- Compute Summary ----------------------
    success_rate = np.mean(all_success_flags)
    avg_reward = np.mean(all_rewards)
    avg_steps = np.mean(all_steps)

    print(f"\n🏁 Evaluation Summary:")
    print(f"   Episodes: {episodes}")
    print(f"   Success rate: {success_rate*100:.1f}%")
    print(f"   Avg reward: {avg_reward:.2f}")
    print(f"   Avg steps: {avg_steps:.1f}")

    # Rolling success rate
    window = 10
    rolling_success = np.convolve(all_success_flags, np.ones(window)/window, mode='valid')

    # Value table
    v_table = np.max(agent.q, axis=-1)
    v_mean = np.mean(v_table, axis=tuple(range(2, v_table.ndim))) if v_table.ndim >= 2 else v_table

    # Mean timelines
    mean_angle = np.nanmean(np.array(all_angle_timelines), axis=0)
    mean_angspeed = np.nanmean(np.array(all_angspeed_timelines), axis=0)
    time_axis = np.arange(len(mean_angle))

    # ---------------------- Plot Summary ----------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("MonsterTruck Q-Learning Evaluation Summary", fontsize=14, weight="bold")

    # --- Total Reward per Episode ---
    ax = axes[0, 0]
    ax.plot(all_rewards, marker='o', alpha=0.7)
    ax.set_title("Total Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True)

    # --- Rolling Success Rate ---
    ax = axes[0, 1]
    ax.plot(rolling_success, color='green')
    ax.set_title(f"Rolling Success Rate (window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True)

    # --- Mean Angle vs Time ---
    ax = axes[1, 0]
    ax.plot(time_axis, mean_angle, color='purple')
    ax.set_title("Mean Angle vs Time (degrees)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Angle from Upright (°)")
    ax.grid(True)

    # --- Value Function Heatmap ---
    ax = axes[1, 1]
    im = ax.imshow(v_mean.T, origin='lower', aspect='auto', cmap='plasma')
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Value Function Heatmap (avg over others)")
    ax.set_xlabel("up_z bins")
    ax.set_ylabel("ang_speed bins")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, "eval_summary.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"✅ Summary figure saved to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    evaluate(episodes=100, max_steps=2000)
