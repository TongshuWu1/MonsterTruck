# evaluate_monstertruck_ypr.py
# Evaluation for MonsterTruckFlipEnvYPR (YPR-based Q-learning)
# Prints last 10 rewards + ?/10 success groups

import os, math, time
import numpy as np
import matplotlib.pyplot as plt
from QLearning_Monstertruck import MonsterTruckFlipEnvYPR, QAgent


def evaluate(episodes=100, max_steps=2000, save_dir="MonsterTruck_eval/YPR_eval"):
    os.makedirs(save_dir, exist_ok=True)

    agent = QAgent()
    if not agent.load():
        print("⚠️ No trained Q-table found. Please train first.")
        return
    agent.eps = 0.0  # greedy policy

    total_success = 0
    rewards, steps_list, success_flags = [], [], []

    for ep in range(1, episodes + 1):
        render = (ep % 20 == 0)
        env = MonsterTruckFlipEnvYPR(render=render, max_steps=max_steps, realtime=render)
        obs = env.reset()
        ep_ret, steps = 0.0, 0
        success_flag = 0

        for _ in range(max_steps):
            a = agent.act(obs)
            obs, r, done, info = env.step(a)
            ep_ret += r
            steps += 1
            if done:
                if info.get("success", False):
                    success_flag = 1
                    total_success += 1
                break

        env.close()
        rewards.append(ep_ret)
        success_flags.append(success_flag)
        steps_list.append(steps)

        if ep % 10 == 0:
            recent_rewards = rewards[-10:]
            recent_success = sum(success_flags[-10:])
            print(f"Eval {ep:3d}/{episodes} | Last 10 rewards: {[round(r, 2) for r in recent_rewards]} | Success {recent_success}/10")

    success_rate = np.mean(success_flags)
    avg_reward = np.mean(rewards)
    print(f"\n🏁 Evaluation Summary:")
    print(f"   Success rate: {success_rate*100:.1f}%")
    print(f"   Avg reward: {avg_reward:.2f}")

    np.save(os.path.join(save_dir, "eval_rewards.npy"), rewards)
    np.save(os.path.join(save_dir, "eval_success.npy"), success_flags)
    print(f"💾 Saved evaluation results to {save_dir}")

    return {"rewards": rewards, "success_flags": success_flags}


if __name__ == "__main__":
    evaluate(episodes=100, max_steps=2000)
