# ===============================================================
# evaluate_carflip_imu_tile.py
# Works with: QLearning_CarFlip_IMU_Tile.py (CarFlipIMUEnv, QAgent)
# Loads tile weights from OUTPUT_DIR/carflip_tile_weights.npy
# Output: carflip_qlearn_output/tile_eval/eval_summary_combined.png
# Panels:
#   (1) φ (deg) vs Time ±std
#   (2) φ̇ (deg/s) vs Time ±std
#   (3) Throttle vs Time ±std
#   (4) Value Heatmap (φ-bin × Action) + Trajectory Overlay
# ===============================================================

import os, math
import numpy as np
import matplotlib.pyplot as plt

from QLearning_Monstertruck import CarFlipIMUEnv, QAgent, OUTPUT_DIR


def phi_to_bin(phi_deg, n_phi_bins):
    """
    Map φ ∈ [-180, 180] to bin index [0..n_phi_bins-1].
    """
    phi = float(phi_deg)
    phi = max(-180.0, min(180.0, phi))
    # map [-180,180] → [0, n-1]
    return int(np.clip((phi + 180.0) / 360.0 * (n_phi_bins - 1), 0, n_phi_bins - 1))


# ---------------------------------------------------------------
# Plot everything into ONE combined PNG
# ---------------------------------------------------------------
def plot_eval_summary(phi_histories, rate_histories, throttle_histories,
                      agent, save_dir="carflip_qlearn_output/tile_eval"):

    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("CarFlip IMU Tile-Q — Evaluation Summary", fontsize=14, fontweight="bold")

    # 1) φ vs Time ± std
    if phi_histories:
        min_len = min(len(p) for p in phi_histories)
        P = np.array([p[:min_len] for p in phi_histories])
        mean, std = P.mean(axis=0), P.std(axis=0)
        axs[0, 0].plot(mean, lw=2)
        axs[0, 0].fill_between(range(min_len), mean - std, mean + std, alpha=0.2)
    axs[0, 0].set_title("Flip Angle φ vs Time")
    axs[0, 0].set_xlabel("Time Step")
    axs[0, 0].set_ylabel("φ (deg)")
    axs[0, 0].grid(alpha=0.3)

    # 2) φ̇ vs Time ± std
    if rate_histories:
        min_len_v = min(len(v) for v in rate_histories)
        V = np.array([v[:min_len_v] for v in rate_histories])
        mean, std = V.mean(axis=0), V.std(axis=0)
        axs[0, 1].plot(mean, lw=2)
        axs[0, 1].fill_between(range(min_len_v), mean - std, mean + std, alpha=0.2)
    axs[0, 1].set_title("Angular Velocity φ̇ vs Time")
    axs[0, 1].set_xlabel("Time Step")
    axs[0, 1].set_ylabel("φ̇ (deg/s)")
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

    # 4) Value Heatmap (φ-bin × Action) + trajectory overlay
    if agent is not None:
        n_actions = agent.n_actions
        n_phi_bins = 121   # resolution for visualization grid

        phi_grid = np.linspace(-180.0, 180.0, n_phi_bins)
        Q_grid = np.zeros((n_phi_bins, n_actions), dtype=float)
        for i, phi in enumerate(phi_grid):
            q_vals = agent._q_values(np.array([phi, 0.0], dtype=np.float32))  # fix φ̇=0
            Q_grid[i, :] = q_vals

        im = axs[1, 1].imshow(
            Q_grid.T,
            origin="lower",
            aspect="auto",
            cmap="plasma",
            extent=[0, n_phi_bins - 1, 0, n_actions - 1],
        )
        axs[1, 1].set_title("Value Function: φ-bin × Action (φ̇ fixed at 0)")
        axs[1, 1].set_xlabel("Angle bin (φ ∈ [-180°, 180°])")
        axs[1, 1].set_ylabel("Action index")
        fig.colorbar(im, ax=axs[1, 1], shrink=0.8)

        # Overlay trajectory from first successful episode (if available)
        if phi_histories and throttle_histories:
            phi_path = np.array(phi_histories[0])         # deg
            throttle_path = np.array(throttle_histories[0])  # throttle values
            act_vals = agent.actions.astype(float)

            # map throttle to nearest action index
            act_idxs = np.array(
                [int(np.argmin(np.abs(act_vals - u))) for u in throttle_path],
                dtype=int,
            )

            # map φ to φ-bin index
            x_bins = np.array([phi_to_bin(phi, n_phi_bins) for phi in phi_path], dtype=int)

            t = np.linspace(0, 1, len(x_bins))
            for i in range(len(x_bins) - 1):
                axs[1, 1].plot(
                    [x_bins[i], x_bins[i + 1]],
                    [act_idxs[i], act_idxs[i + 1]],
                    color=plt.cm.cividis(t[i]),
                    lw=2,
                    alpha=0.95,
                )
            axs[1, 1].scatter(x_bins[0], act_idxs[0], c="lime", s=50, label="Start", zorder=3)
            axs[1, 1].scatter(x_bins[-1], act_idxs[-1], c="red", s=50, label="End", zorder=3)
            axs[1, 1].legend(loc="upper right", fontsize=8)
    else:
        axs[1, 1].set_title("No agent / weights available")

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
def evaluate(
    episodes=100,
    max_steps=1500,
    save_dir="carflip_qlearn_output/tile_eval",
    render_every=10,
):
    os.makedirs(save_dir, exist_ok=True)

    # Instantiate agent with same config as training and load weights
    agent = QAgent()
    if not agent.load():  # looks for OUTPUT_DIR/carflip_tile_weights.npy
        print("⚠️ Could not load tile weights. Train first to generate carflip_tile_weights.npy.")
        return

    rewards, success_flags = [], []
    phi_histories, rate_histories, throttle_histories = [], [], []

    for ep in range(1, episodes + 1):
        render = (ep % render_every == 0)
        env = CarFlipIMUEnv(
            render=render,
            realtime=render,
            max_steps=max_steps,
            frame_skip=10,
            seed=ep,  # different seeds for variety
        )
        obs = env.reset()

        ep_ret = 0.0
        success_flag = 0

        throttle_trace, phi_trace, rate_trace = [], [], []

        for _ in range(max_steps):
            a_idx = agent.act_greedy(obs)
            throttle_val = float(agent.actions[a_idx])
            throttle_trace.append(throttle_val)

            phi_deg = float(obs[0])
            phi_rate_deg = float(obs[1])
            phi_trace.append(phi_deg)
            rate_trace.append(phi_rate_deg)

            next_obs, r, done, info = env.step(throttle_val)
            ep_ret += r
            obs = next_obs

            if done:
                if info.get("success", False):
                    success_flag = 1
                break

        env.close()
        rewards.append(ep_ret)
        success_flags.append(success_flag)

        # store only successful runs for mean curves
        if success_flag:
            phi_histories.append(phi_trace)
            rate_histories.append(rate_trace)
            throttle_histories.append(throttle_trace)

        if ep % 10 == 0:
            recent_rewards = rewards[-10:]
            recent_success = sum(success_flags[-10:])
            print(
                f"Eval {ep:3d}/{episodes} | "
                f"Last 10 rewards: {[round(r, 2) for r in recent_rewards]} | "
                f"Success {recent_success}/10"
            )

    # Summary + Plot
    sr = np.mean(success_flags) if success_flags else 0.0
    avg_r = np.mean(rewards) if rewards else 0.0
    print("\n🏁 Evaluation Summary:")
    print(f"   Success rate: {sr*100:.1f}%")
    print(f"   Avg reward: {avg_r:.2f}")

    np.save(os.path.join(save_dir, "eval_rewards.npy"), rewards)
    np.save(os.path.join(save_dir, "eval_success.npy"), success_flags)

    plot_eval_summary(
        phi_histories,
        rate_histories,
        throttle_histories,
        agent,
        save_dir=save_dir,
    )

    print(f"💾 Saved evaluation results to {save_dir}")
    return {"rewards": rewards, "success_flags": success_flags}


# ---------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------
if __name__ == "__main__":
    evaluate(episodes=100, max_steps=1500, render_every=10)
