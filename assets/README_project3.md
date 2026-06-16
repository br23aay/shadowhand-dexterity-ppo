# Graceful Under Pressure — Tactile Sensor Noise Robustness in Shadow Hand PPO

**Project 3 | Sim-to-Real Transfer Research | Bharadwaj Rachuri**

A systematic empirical study of how tactile sensor failures and noise degrade a trained PPO policy for dexterous in-hand pen manipulation using the Shadow Hand.

---

## Research Question

When tactile sensors fail or degrade during real-world deployment, does a PPO policy fail gracefully or catastrophically? Which sensors are critical?

---

## Key Findings

| Finding | Result |
|---|---|
| **Robustness threshold** | Policy tolerates Gaussian noise up to std=0.30 with <2 reward unit degradation |
| **Critical failure cliff** | At std=0.50, reward degrades by **622 units** (142% of baseline) |
| **Dropout robustness** | Policy tolerates up to 50% random dropout; 80% causes 615 unit degradation |
| **Bias offset** | Most robust noise type — even bias=0.30 causes <2 units degradation |
| **Thumb is critical** | Thumb sensor failure → 620 unit degradation; all other fingers → 0 degradation |

---

## Results

### Gaussian Sensor Noise
![Gaussian Noise](results/figures/gaussian_noise.png)

Clear robustness cliff between std=0.30 and std=0.50. Policy is highly robust to low-intensity noise.

### Partial Finger Failure
![Finger Failure](results/figures/finger_failure.png)

Thumb sensor is disproportionately critical. Losing any other finger sensor has zero measurable impact.

---

## Noise Conditions Tested

| Noise Type | Parameters | Finding |
|---|---|---|
| Gaussian noise | std = 0.0, 0.05, 0.1, 0.2, 0.3, 0.5 | Cliff at std=0.5 |
| Sensor dropout | rate = 0.0, 0.1, 0.2, 0.3, 0.5, 0.8 | Cliff at rate=0.8 |
| Bias offset | 0.0, 0.05, 0.1, 0.2, 0.3 | Robust throughout |
| Partial finger failure | Index, Middle, Ring, Little, Thumb | Thumb critical only |

---

## Methodology

1. Train PPO policy on Shadow Hand pen spinning task (75,000 timesteps, CPU)
2. Evaluate baseline policy across 10 episodes
3. Inject structured noise into tactile sensor observations at inference time
4. Measure reward degradation and success rate under each noise condition
5. Generate degradation curves and finger criticality analysis

**Noise injection is applied at inference only** — the policy weights are not retrained. This directly simulates real-world sensor degradation during deployment.

---

## Sim-to-Real Relevance

This study frames tactile sensor dropout as **domain randomisation** — a key technique for sim-to-real transfer. The findings suggest:

- Tactile sensors with noise std < 0.30 can be tolerated without policy retraining
- The thumb fingertip sensor requires hardware redundancy for robust deployment
- Bias calibration drift is the least harmful failure mode

---

## Project Structure

```
shadowhand-dexterity-ppo/
├── env/
│   └── pen_spin_env.py         ← MuJoCo gymnasium environment (obs=43, act=7)
├── scripts/
│   ├── train.py                ← PPO training script
│   └── evaluate.py             ← Policy evaluation
├── assets/
│   └── shadow_right_hand.xml   ← Self-contained MuJoCo scene
├── models/
│   └── ppo_pen_spin_success.zip ← Trained PPO policy
├── noise_robustness.py         ← Main experiment script
├── plot_results.py             ← Figure generation
└── results/
    ├── noise_robustness_results.csv
    └── figures/
        ├── gaussian_noise.png
        ├── dropout_noise.png
        ├── bias_offset.png
        └── finger_failure.png
```

---

## Quickstart

```bash
# Create virtual environment
python -m venv shadowhand_env
shadowhand_env\Scripts\activate   # Windows
source shadowhand_env/bin/activate # Linux/Mac

# Install dependencies
pip install stable-baselines3==2.5.0 gymnasium==1.0.0 numpy==2.2.2
pip install torch==2.6.0 mujoco==3.1.6 matplotlib==3.8.0 tensorboard

# Train policy (~2 hours CPU)
python scripts/train.py

# Run noise robustness experiment (~15 min)
python noise_robustness.py

# Generate figures
python plot_results.py
```

---

## Skills Evidenced

- **Reinforcement Learning** — PPO training, Stable-Baselines3, MuJoCo
- **Sim-to-Real Transfer** — domain randomisation via sensor noise injection
- **Experimental Design** — systematic ablation across 5 noise types, 20 conditions
- **Scientific Visualisation** — publication-quality degradation curves
- **Python** — gymnasium environments, numpy, matplotlib

---

## Related Work

This project extends the findings from:

> Rachuri, B. & Faria, D.R. (2025). *Reinforcement Learning for Robot Dexterous In-Hand Manipulation of Objects (Shadow Hand)*. IJRES Vol.13, Issue 6, pp.164-183. IF: 7.52.

The published paper established the PPO baseline. This study examines deployment robustness — a critical gap between simulation and real-world use.

---

## Author

**Bharadwaj Rachuri** — ML & AI Engineer
[br23aay.github.io](https://br23aay.github.io) · [github.com/br23aay](https://github.com/br23aay)
