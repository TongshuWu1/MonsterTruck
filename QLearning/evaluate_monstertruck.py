# ===============================================================
# evaluate_monstertruck_ypr.py
# Works with: QLearning_Monstertruck_YPR.py (agent.q, env.actions)
# Output: MonsterTruck_eval/YPR_eval/eval_summary_combined.png
# Panels:
#   (1) Angle vs Time ±std
#   (2) Angular Velocity vs Time ±std
#   (3) Throttle vs Time ±std
#   (4) Value Heatmap (AngleBin × Action) + Trajectory Overlay
# ===============================================================

import os, math, numpy as np
import matplotlib.pyplot as plt
from QLearning_Monstertruck import MonsterTruckFlipEnvYPR, QAgent

# ---------------------------
# Small helpers
# ---------------------------
def upz_to_degrees(u):
    u = float(np.clip(u, -1.0, 1.0))
    return math.degrees(math.acos(u))

def angle_deg_to_upz_bin(angle_deg, n_upz_bins):
    """Map angle in [0,180] to bin index [0..n_upz_bins-1] using cos(angle)."""
    up_z = math.cos(math.radians(float(angle_deg)))
    if n_upz_bins <= 1:
        return 0
    # map [-1,1] -> [0, n-1]
    return int(np.clip((up_z + 1.0) * 0.5 * (n_upz_bins - 1), 0, n_upz_bins - 1))

# ---------------------------------------------------------------
# Plot everything into ONE combined PNG
# ---------------------------------------------------------------
def plot_eval_summary(angle_histories, angvel_histories, throttle_histories,
                      q_table, action_values, save_dir="MonsterTruck_eval/YPR_eval"):

    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("MonsterTruck YPR Q-Learning — Evaluation Summary", fontsize=14, fontweight="bold")

    # 1) Angle vs Time ± std
    if angle_histories:
        min_len = min(len(a) for a in angle_histories)
        A = np.array([a[:min_len] for a in angle_histories])
        mean, std = A.mean(axis=0), A.std(axis=0)
        axs[0, 0].plot(mean, lw=2)
        axs[0, 0].fill_between(range(min_len), mean - std, mean + std, alpha=0.2)
    axs[0, 0].set_title("Angle vs Time (°)")
    axs[0, 0].set_xlabel("Time Step")
    axs[0, 0].set_ylabel("Angle from Upright (°)")
    axs[0, 0].grid(alpha=0.3)

    # 2) Angular Velocity vs Time ± std
    if angvel_histories:
        min_len_v = min(len(v) for v in angvel_histories)
        V = np.array([v[:min_len_v] for v in angvel_histories])
        mean, std = V.mean(axis=0), V.std(axis=0)
        axs[0, 1].plot(mean, lw=2)
        axs[0, 1].fill_between(range(min_len_v), mean - std, mean + std, alpha=0.2)
    axs[0, 1].set_title("Angular Velocity vs Time (rad/s)")
    axs[0, 1].set_xlabel("Time Step")
    axs[0, 1].set_ylabel("Angular Velocity (rad/s)")
    axs[0, 1].grid(alpha=0.3)

    # 3) Throttle vs Time ± std
    if throttle_histories:
        min_len_t = min(len(t) for t in throttle_histories)
        T = np.array([t[:min_len_t] for t in throttle_histories])
        mean, std = T.mean(axis=0), T.std(axis=0)
        axs[1, 0].plot(mean, lw=2)
        axs[1, 0].fill_between(range(min_len_t), mean - std, mean + std, alpha=0.2)
    axs[1, 0].set_title("Throttle vs Time")
    axs[1, 0].set_xlabel("Time Step")
    axs[1, 0].set_ylabel("Throttle")
    axs[1, 0].axhline(0, color="black", lw=1, ls="--")
    axs[1, 0].grid(alpha=0.3)

    # 4) Value Heatmap (AngleBin × Action) + trajectory
    if q_table is not None and q_table.ndim >= 2:
        # q shape: (n_upz_bins, other_state_bins..., n_actions)
        n_actions = q_table.shape[-1]
        n_upz_bins = q_table.shape[0]
        # reduce all state dims except up_z (axis 0)
        if q_table.ndim > 2:
            Q = q_table.mean(axis=tuple(range(1, q_table.ndim - 1)))
        else:
            Q = q_table
        # Q shape now: (n_upz_bins, n_actions)
        im = axs[1, 1].imshow(Q.T, origin="lower", aspect="auto", cmap="plasma")
        axs[1, 1].set_title("Value Function: AngleBin × Action (avg over other dims)")
        axs[1, 1].set_xlabel("Angle bin (upright → inverted)")
        axs[1, 1].set_ylabel("Action index")
        fig.colorbar(im, ax=axs[1, 1], shrink=0.8)

        # Overlay trajectory from first successful episode (if available)
        if angle_histories and throttle_histories:
            angle_path = np.array(angle_histories[0])  # degrees
            throttle_path = np.array(throttle_histories[0])  # raw throttle values

            # Map throttle values to nearest action index
            action_values = np.asarray(action_values, dtype=float)
            idxs = np.array([int(np.argmin(np.abs(action_values - v))) for v in throttle_path], dtype=int)

            # Map angle degrees to up_z bins
            x_bins = np.array([angle_deg_to_upz_bin(a, n_upz_bins) for a in angle_path], dtype=int)

            # Color by time progression
            t = np.linspace(0, 1, len(x_bins))
            for i in range(len(x_bins) - 1):
                axs[1, 1].plot([x_bins[i], x_bins[i + 1]],
                               [idxs[i], idxs[i + 1]],
                               color=plt.cm.cividis(t[i]), lw=2, alpha=0.95)
            axs[1, 1].scatter(x_bins[0], idxs[0], c="lime", s=50, label="Start", zorder=3)
            axs[1, 1].scatter(x_bins[-1], idxs[-1], c="red", s=50, label="End", zorder=3)
            axs[1, 1].legend(loc="upper right", fontsize=8)
    else:
        axs[1, 1].set_title("No Q-table available")

    # Save one combined PNG
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig = plt.gcf()
    fig.canvas.draw()
    fig_path = os.path.abspath(os.path.join(save_dir, "eval_summary_combined.png"))
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"📊 Saved all graphs together → {fig_path}")

# ---------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------
def evaluate(episodes=100, max_steps=1500, save_dir="MonsterTruck_eval/YPR_eval", render_every=20):
    os.makedirs(save_dir, exist_ok=True)

    # Load agent (uses agent.q and qtable_ypr.npy by default)
    agent = QAgent()
    if not agent.load():  # loads qtable_ypr.npy, qmeta_ypr.json
        print("⚠️ No trained Q-table found (qtable_ypr.npy). Please train first.")
        return

    # Debug: show q-table stats so you know it's loaded
    print(f"Q shape: {agent.q.shape}, mean={agent.q.mean():.4f}, min={agent.q.min():.4f}, max={agent.q.max():.4f}")

    rewards, success_flags = [], []
    angle_histories, angvel_histories, throttle_histories = [], [], []

    for ep in range(1, episodes + 1):
        render = (ep % render_every == 0)
        env = MonsterTruckFlipEnvYPR(render=render, max_steps=max_steps, realtime=render)
        obs = env.reset()

        # env parameters needed for ang vel estimate & mapping
        dt, frameskip = env.dt, env.frame_skip
        act_vals = env.actions  # array of throttle values

        ep_ret = 0.0
        success_flag = 0

        throttle_trace, angle_trace, angvel_trace = [], [], []
        prev_roll, prev_pitch = obs[1], obs[2]

        for _ in range(max_steps):
            a_idx = agent.act_greedy(obs)
            throttle_val = float(act_vals[a_idx])
            throttle_trace.append(throttle_val)

            # angle BEFORE step (based on current obs)
            u = float(obs[0])  # upright cosine provided by env
            angle_trace.append(upz_to_degrees(u))

            # step
            next_obs, r, done, info = env.step(a_idx)
            ep_ret += r

            # approximate angular velocity from roll/pitch deltas
            roll, pitch = next_obs[1], next_obs[2]
            droll, dpitch = roll - prev_roll, pitch - prev_pitch
            prev_roll, prev_pitch = roll, pitch
            ang_speed = math.sqrt(droll * droll + dpitch * dpitch) / max(1e-9, (dt * frameskip))
            angvel_trace.append(ang_speed)

            obs = next_obs
            if done:
                if info.get("success", False):
                    success_flag = 1
                break

        env.close()
        rewards.append(ep_ret)
        success_flags.append(success_flag)

        # keep only successful runs for the mean curves (your earlier preference)
        if success_flag:
            angle_histories.append(angle_trace)
            angvel_histories.append(angvel_trace)
            throttle_histories.append(throttle_trace)

        if ep % 10 == 0:
            recent_rewards = rewards[-10:]
            recent_success = sum(success_flags[-10:])
            print(f"Eval {ep:3d}/{episodes} | "
                  f"Last 10 rewards: {[round(r, 2) for r in recent_rewards]} | "
                  f"Success {recent_success}/10")

    # Summary + Plot
    sr = np.mean(success_flags) if success_flags else 0.0
    avg_r = np.mean(rewards) if rewards else 0.0
    print("\n🏁 Evaluation Summary:")
    print(f"   Success rate: {sr*100:.1f}%")
    print(f"   Avg reward: {avg_r:.2f}")

    np.save(os.path.join(save_dir, "eval_rewards.npy"), rewards)
    np.save(os.path.join(save_dir, "eval_success.npy"), success_flags)

    plot_eval_summary(angle_histories, angvel_histories, throttle_histories,
                      agent.q,  # <-- use agent.q (not agent.Q)
                      action_values=act_vals,
                      save_dir=save_dir)

    print(f"💾 Saved evaluation results to {save_dir}")
    return {"rewards": rewards, "success_flags": success_flags}

# ---------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------
if __name__ == "__main__":
    evaluate(episodes=100, max_steps=1500, render_every=10)
