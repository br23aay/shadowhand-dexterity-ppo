"""
plot_results.py
---------------
Generates publication-quality figures from noise robustness experiment results.

Produces:
    1. Reward degradation curve (Gaussian noise)
    2. Reward degradation curve (Dropout noise)
    3. Bias offset sensitivity plot
    4. Partial finger failure bar chart
    5. Summary heatmap — all conditions

Usage:
    python plot_results.py
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_FILE = "results/noise_robustness_results.csv"
FIGURES_DIR = "results/figures"


def load_results():
    """Load results CSV into a list of dicts."""
    results = []
    with open(RESULTS_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "noise_type": row["noise_type"],
                "noise_param": float(row["noise_param"]),
                "mean_reward": float(row["mean_reward"]),
                "std_reward": float(row["std_reward"]),
                "success_rate": float(row["success_rate"]),
                "mean_angle_rad": float(row["mean_angle_rad"]),
            })
    return results


def filter_by_type(results, noise_type):
    """Filter results by noise type."""
    return [r for r in results if r["noise_type"] == noise_type]


def plot_degradation_curve(results, noise_type, param_label, title, filename):
    """Plot reward vs noise parameter with error bars."""
    baseline = filter_by_type(results, "baseline")[0]
    noisy = filter_by_type(results, noise_type)

    params = [0.0] + [r["noise_param"] for r in noisy]
    rewards = [baseline["mean_reward"]] + [r["mean_reward"] for r in noisy]
    stds = [baseline["std_reward"]] + [r["std_reward"] for r in noisy]
    success = [baseline["success_rate"]] + [r["success_rate"] for r in noisy]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Reward curve
    ax1.errorbar(params, rewards, yerr=stds, fmt="o-",
                 color="#00bcd4", linewidth=2, markersize=6,
                 capsize=4, label="Mean Reward")
    ax1.axhline(y=baseline["mean_reward"], color="gray",
                linestyle="--", alpha=0.5, label="Baseline")
    ax1.set_ylabel("Episode Reward", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Success rate
    ax2.plot(params, success, "s-", color="#9c27b0",
             linewidth=2, markersize=6, label="Success Rate")
    ax2.axhline(y=baseline["success_rate"], color="gray",
                linestyle="--", alpha=0.5)
    ax2.set_xlabel(param_label, fontsize=11)
    ax2.set_ylabel("Success Rate", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_finger_failure(results):
    """Bar chart of performance under each finger failure."""
    baseline = filter_by_type(results, "baseline")[0]
    finger_names = ["Index", "Middle", "Ring", "Little", "Thumb"]
    finger_types = [f"partial_failure_{n.lower()}" for n in finger_names]

    rewards = []
    success = []
    for ftype in finger_types:
        r = filter_by_type(results, ftype)
        if r:
            rewards.append(r[0]["mean_reward"])
            success.append(r[0]["success_rate"])
        else:
            rewards.append(0.0)
            success.append(0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Partial Finger Failure — Performance Impact",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(finger_names))
    colors = ["#00bcd4", "#4caf50", "#ff9800", "#e91e63", "#9c27b0"]

    bars1 = ax1.bar(x, rewards, color=colors, alpha=0.85, edgecolor="white")
    ax1.axhline(y=baseline["mean_reward"], color="gray",
                linestyle="--", label=f"Baseline: {baseline['mean_reward']:.0f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(finger_names)
    ax1.set_ylabel("Mean Episode Reward")
    ax1.set_title("Reward by Failed Finger")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    bars2 = ax2.bar(x, success, color=colors, alpha=0.85, edgecolor="white")
    ax2.axhline(y=baseline["success_rate"], color="gray",
                linestyle="--", label=f"Baseline: {baseline['success_rate']:.2f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(finger_names)
    ax2.set_ylabel("Success Rate")
    ax2.set_title("Success Rate by Failed Finger")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "finger_failure.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: finger_failure.png")


def plot_summary_table(results):
    """Print a clean summary table to console."""
    baseline = filter_by_type(results, "baseline")[0]
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY — Noise Robustness Study")
    print("=" * 65)
    print(f"{'Condition':<35} {'Reward':>10} {'Success':>10} {'Degradation':>12}")
    print("-" * 65)

    print(f"{'Baseline (no noise)':<35} "
          f"{baseline['mean_reward']:>10.1f} "
          f"{baseline['success_rate']:>10.3f} "
          f"{'—':>12}")

    categories = [
        ("Gaussian noise", "gaussian"),
        ("Dropout noise", "dropout"),
        ("Bias offset", "bias"),
    ]

    for label, noise_type in categories:
        noisy = filter_by_type(results, noise_type)
        if noisy:
            worst = min(noisy, key=lambda x: x["mean_reward"])
            deg = baseline["mean_reward"] - worst["mean_reward"]
            print(f"{label + ' (worst)':<35} "
                  f"{worst['mean_reward']:>10.1f} "
                  f"{worst['success_rate']:>10.3f} "
                  f"{deg:>11.1f}")

    finger_names = ["index", "middle", "ring", "little", "thumb"]
    for name in finger_names:
        r = filter_by_type(results, f"partial_failure_{name}")
        if r:
            deg = baseline["mean_reward"] - r[0]["mean_reward"]
            print(f"{'Finger failure: ' + name:<35} "
                  f"{r[0]['mean_reward']:>10.1f} "
                  f"{r[0]['success_rate']:>10.3f} "
                  f"{deg:>11.1f}")

    print("=" * 65)


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file not found: {RESULTS_FILE}")
        print("Run noise_robustness.py first.")
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)
    results = load_results()

    print("Generating figures...")

    plot_degradation_curve(
        results, "gaussian", "Gaussian Noise Std Dev",
        "Gaussian Sensor Noise — Reward & Success Rate Degradation",
        "gaussian_noise.png"
    )

    plot_degradation_curve(
        results, "dropout", "Sensor Dropout Rate",
        "Tactile Sensor Dropout — Reward & Success Rate Degradation",
        "dropout_noise.png"
    )

    plot_degradation_curve(
        results, "bias", "Bias Offset Value",
        "Calibration Bias Offset — Reward & Success Rate Degradation",
        "bias_offset.png"
    )

    plot_finger_failure(results)
    plot_summary_table(results)

    print(f"\nAll figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
