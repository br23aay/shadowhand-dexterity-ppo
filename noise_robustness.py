"""
noise_robustness.py
-------------------
Project 3: Graceful Under Pressure
Tactile Sensor Noise Robustness Study — Shadow Hand PPO Policy

Injects structured noise into tactile sensor observations at inference time
and measures policy performance degradation.

Noise types studied:
    1. Gaussian noise     — random additive noise to all sensors
    2. Dropout noise      — random sensor channels zeroed out
    3. Bias offset        — systematic calibration drift on all sensors
    4. Stuck-at fault     — one or more sensors fixed at a constant value
    5. Partial failure    — one finger's sensors completely disabled

Each condition is evaluated across 10 episodes.
Results saved to results/noise_robustness_results.csv

Usage:
    python noise_robustness.py

Reference:
    Rachuri, B. & Faria, D.R. (2025). IJRES Vol.13, Issue 6, pp.164-183.
"""

import os
import csv
import numpy as np
from stable_baselines3 import PPO
from env.pen_spin_env import PenSpinEnv

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/ppo_pen_spin_success.zip"
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "noise_robustness_results.csv")
NUM_EPISODES = 10
MAX_STEPS = 1000
TOUCH_OBS_START = 20   # Index in observation where touch sensors begin
TOUCH_OBS_END = 25     # 5 touch sensor channels

# Noise parameter sweep values
GAUSSIAN_STDS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
BIAS_OFFSETS = [0.0, 0.05, 0.1, 0.2, 0.3]
STUCK_AT_VALUES = [0.0, 0.5, 1.0]
PARTIAL_FAILURE_FINGERS = [0, 1, 2, 3, 4]  # which finger fails


# ── Noise injection functions ─────────────────────────────────────────────────

def inject_gaussian_noise(obs, std):
    """Add Gaussian noise to tactile sensor channels."""
    obs = obs.copy()
    noise = np.random.normal(0, std, size=5).astype(np.float32)
    obs[TOUCH_OBS_START:TOUCH_OBS_END] += noise
    obs[TOUCH_OBS_START:TOUCH_OBS_END] = np.clip(
        obs[TOUCH_OBS_START:TOUCH_OBS_END], 0.0, 1.0
    )
    return obs


def inject_dropout_noise(obs, dropout_rate):
    """Randomly zero out tactile sensor channels."""
    obs = obs.copy()
    mask = np.random.binomial(1, 1 - dropout_rate, size=5).astype(np.float32)
    obs[TOUCH_OBS_START:TOUCH_OBS_END] *= mask
    return obs


def inject_bias_offset(obs, bias):
    """Apply systematic calibration drift to all tactile sensors."""
    obs = obs.copy()
    obs[TOUCH_OBS_START:TOUCH_OBS_END] += bias
    obs[TOUCH_OBS_START:TOUCH_OBS_END] = np.clip(
        obs[TOUCH_OBS_START:TOUCH_OBS_END], 0.0, 1.0
    )
    return obs


def inject_stuck_at(obs, stuck_value, finger_idx=0):
    """Fix one sensor channel at a constant value (stuck-at fault)."""
    obs = obs.copy()
    obs[TOUCH_OBS_START + finger_idx] = stuck_value
    return obs


def inject_partial_failure(obs, finger_idx):
    """Completely disable one finger's tactile sensor."""
    obs = obs.copy()
    obs[TOUCH_OBS_START + finger_idx] = 0.0
    return obs


def no_noise(obs, *args):
    """Baseline — no noise injection."""
    return obs.copy()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_policy(model, env, noise_fn, noise_param, num_episodes=NUM_EPISODES):
    """
    Run num_episodes with noise injection and return performance metrics.

    Returns:
        {
            mean_reward, std_reward,
            mean_steps, std_steps,
            mean_angle, std_angle,
            success_rate
        }
    """
    rewards = []
    steps = []
    final_angles = []
    successes = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0.0
        step = 0
        final_angle = 0.0

        while not done and step < MAX_STEPS:
            # Inject noise into observation before prediction
            noisy_obs = noise_fn(obs, noise_param)
            action, _ = model.predict(noisy_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            final_angle = info.get("pen_angle", 0.0)

        import math
        success = abs(final_angle - math.pi) < 0.1
        rewards.append(total_reward)
        steps.append(step)
        final_angles.append(abs(final_angle))
        successes.append(int(success))

    return {
        "mean_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "mean_steps": round(float(np.mean(steps)), 1),
        "std_steps": round(float(np.std(steps)), 1),
        "mean_angle_rad": round(float(np.mean(final_angles)), 4),
        "std_angle_rad": round(float(np.std(final_angles)), 4),
        "success_rate": round(float(np.mean(successes)), 3),
    }


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment():
    """Run all noise conditions and save results."""
    print("=" * 65)
    print("Project 3 — Tactile Sensor Noise Robustness Study")
    print("Shadow Hand PPO Policy | Graceful Under Pressure")
    print("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load trained policy
    print(f"\nLoading model: {MODEL_PATH}")
    env = PenSpinEnv(render=False)
    model = PPO.load(MODEL_PATH, env=env)
    print("Model loaded successfully.")

    results = []

    # ── 1. Baseline (no noise) ────────────────────────────────────────────────
    print("\n[1/5] Baseline (no noise)...")
    metrics = evaluate_policy(model, env, no_noise, 0.0)
    results.append({
        "noise_type": "baseline",
        "noise_param": 0.0,
        **metrics
    })
    print(f"  Mean reward: {metrics['mean_reward']} | "
          f"Success rate: {metrics['success_rate']}")

    # ── 2. Gaussian noise ─────────────────────────────────────────────────────
    print("\n[2/5] Gaussian noise sweep...")
    for std in GAUSSIAN_STDS[1:]:  # skip 0.0 (already done as baseline)
        metrics = evaluate_policy(model, env, inject_gaussian_noise, std)
        results.append({
            "noise_type": "gaussian",
            "noise_param": std,
            **metrics
        })
        print(f"  std={std:.2f} | Mean reward: {metrics['mean_reward']} | "
              f"Success rate: {metrics['success_rate']}")

    # ── 3. Dropout noise ──────────────────────────────────────────────────────
    print("\n[3/5] Dropout noise sweep...")
    for rate in DROPOUT_RATES[1:]:
        metrics = evaluate_policy(model, env, inject_dropout_noise, rate)
        results.append({
            "noise_type": "dropout",
            "noise_param": rate,
            **metrics
        })
        print(f"  rate={rate:.2f} | Mean reward: {metrics['mean_reward']} | "
              f"Success rate: {metrics['success_rate']}")

    # ── 4. Bias offset ────────────────────────────────────────────────────────
    print("\n[4/5] Bias offset sweep...")
    for bias in BIAS_OFFSETS[1:]:
        metrics = evaluate_policy(model, env, inject_bias_offset, bias)
        results.append({
            "noise_type": "bias",
            "noise_param": bias,
            **metrics
        })
        print(f"  bias={bias:.2f} | Mean reward: {metrics['mean_reward']} | "
              f"Success rate: {metrics['success_rate']}")

    # ── 5. Partial finger failure ─────────────────────────────────────────────
    print("\n[5/5] Partial finger failure...")
    finger_names = ["index", "middle", "ring", "little", "thumb"]
    for finger_idx in PARTIAL_FAILURE_FINGERS:
        metrics = evaluate_policy(
            model, env, inject_partial_failure, finger_idx
        )
        results.append({
            "noise_type": f"partial_failure_{finger_names[finger_idx]}",
            "noise_param": finger_idx,
            **metrics
        })
        print(f"  finger={finger_names[finger_idx]} | "
              f"Mean reward: {metrics['mean_reward']} | "
              f"Success rate: {metrics['success_rate']}")

    env.close()

    # ── Save results ──────────────────────────────────────────────────────────
    fieldnames = [
        "noise_type", "noise_param",
        "mean_reward", "std_reward",
        "mean_steps", "std_steps",
        "mean_angle_rad", "std_angle_rad",
        "success_rate"
    ]

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Total conditions evaluated: {len(results)}")
    print("\nBaseline vs worst degradation:")
    baseline = results[0]
    worst = min(results[1:], key=lambda x: x["mean_reward"])
    print(f"  Baseline reward:  {baseline['mean_reward']}")
    print(f"  Worst condition:  {worst['noise_type']} "
          f"(param={worst['noise_param']}) -> {worst['mean_reward']}")
    degradation = baseline['mean_reward'] - worst['mean_reward']
    print(f"  Degradation:      {degradation:.2f} reward units")

    return results


if __name__ == "__main__":
    run_experiment()
