# 🤖 ShadowHand Dexterity PPO

> **Reinforcement Learning for Dexterous In-Hand Object Manipulation**  
> MSc Final Project — University of Hertfordshire, 2025  
> Supervisor: Dr. Diego Resende Faria

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.x-green.svg)](https://mujoco.org)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-PPO-orange.svg)](https://stable-baselines3.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Overview

This project trains a **Shadow Hand robotic platform** to grasp and rotate a pen by 180° using **Proximal Policy Optimization (PPO)** inside a custom **MuJoCo** simulation environment.

The Shadow Hand has **24 degrees of freedom** — one of the most complex robotic hand platforms in the world. A professor at the University of Hertfordshire advised this project would typically require **6–12 months**. It was completed in **3 months**, and the results were published as part of the MSc programme.

**The core challenge:** Teaching a 24-DoF robotic hand to coordinate all five fingers in sequence to grip and rotate a pen to a target angle of π radians (~180°) — without dropping it.

---

## 📊 Key Results

| Metric | Value |
|---|---|
| **Target Rotation** | π radians (180°) |
| **Achieved Rotation** | 3.058 radians (~175°) |
| **Margin of Error** | ~2.6% |
| **Peak Episode Reward** | 18,007.53 |
| **Mean Episode Reward** | 32,600 |
| **Touch Sensor Accuracy** | >90% |
| **Episode Length (converged)** | ~960 steps |
| **Total Training Timesteps** | 70,000+ |

### Reproducibility — 3 Independent Runs (Different Seeds)

| Run | Final Z-Angle (rad) | Episode Reward |
|---|---|---|
| Run 1 | 3.058 | 18,007.53 |
| Run 2 | 3.172 | 19,000.12 |
| Run 3 | 3.123 | 17,879.34 |
| **Mean ± Std** | **3.118 ± 0.047** | **18,295 ± 487** |

The consistent results across 3 different random seeds confirm the policy is stable and generalizable, not just a lucky training run.

---

## 🏗️ Architecture

```
PPO Agent
    │
    ├── Observation Space
    │       ├── Joint positions (24 DoF)
    │       ├── Joint velocities
    │       ├── Tactile sensor readings (5 fingers)
    │       └── Pen orientation (quaternion)
    │
    ├── Action Space
    │       └── Joint torque commands (continuous)
    │
    └── Reward Function
            ├── Phase 1: Initial grasp alignment
            ├── Phase 2: Rotation toward target angle
            ├── Phase 3: Hold and stabilize at 180°
            └── Penalties: Pen drop, angular deviation, drift
```

**Training Progression:**

| Phase | Steps | Avg Reward | Behaviour |
|---|---|---|---|
| Early | 0–10,000 | -4.0 to -0.8 | Random exploration, pen frequently dropped |
| Mid | 10,000–40,000 | 600 → 8,000 | Consistent 143°+ rotations, finger coordination emerging |
| Final | 40,000–70,000+ | 32,600 | Near-180° rotation, stable grip, policy converged |

---

## 🛠️ Tech Stack

- **Simulation:** MuJoCo 2.x with custom Shadow Hand + pen XML model
- **RL Algorithm:** PPO (Proximal Policy Optimization) via Stable-Baselines3
- **Sensors:** Tactile contact sensors across all 5 fingers + framequat orientation sensor
- **Language:** Python 3.9+
- **Tracking:** Custom logging, reward curves, KL divergence, entropy, value/policy loss

---

## 📁 Repository Structure

```
shadowhand-dexterity-ppo/
│
├── assets/              # MuJoCo XML models (Shadow Hand + pen)
├── models/              # Saved PPO policy checkpoints
├── results/             # Training metrics, reward curves, evaluation logs
├── scripts/
│   ├── train.py         # PPO training script
│   ├── evaluate.py      # Policy evaluation and scoring
│   └── visualise.py     # Render trained policy in MuJoCo viewer
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

**requirements.txt** includes:
- `mujoco`
- `stable-baselines3`
- `gymnasium`
- `numpy`
- `torch`

### Train from Scratch

```bash
python scripts/train.py --timesteps 70000 --seed 42
```

### Evaluate a Trained Policy

```bash
python scripts/evaluate.py --model models/ppo_shadowhand_final
```

### Visualise in MuJoCo

```bash
python scripts/visualise.py --model models/ppo_shadowhand_final
```

---

## 📈 Training Insights

**What made the difference:**

1. **Phase-based reward shaping** — Instead of a single reward for reaching 180°, the reward function was broken into phases: grasp → rotate → hold. This prevented reward sparsity and guided the agent step by step.

2. **Tactile sensor integration** — Each finger had contact sensors that only triggered reward when the *correct* finger made contact at the *correct* phase. This enforced coordinated, human-like manipulation.

3. **KL divergence monitoring** — PPO updates were kept conservative (low KL divergence throughout), preventing policy collapse which is a common failure mode in high-DoF environments.

4. **Sim-to-real considerations** — Joint limits, actuator noise, and contact stiffness were tuned to reflect real Shadow Hand behaviour, forming the basis for potential real-world transfer.

---

## 🔬 Research Questions Answered

**RQ1:** Can PPO learn coordinated multi-finger manipulation?  
✅ Yes — the agent achieved 175° rotation with >90% touch sensor accuracy across all 5 fingers.

**RQ2:** What reward design is most effective?  
✅ Phase-specific sensor-driven rewards outperformed simple orientation-based rewards.

**RQ3:** Does tactile feedback improve manipulation quality?  
✅ Yes — sensor-integrated rewards led to more stable grasps and longer episodes compared to orientation-only baselines.

---

## 📄 Publication

This project was submitted and accepted as a Final Project Report for the **MSc in Artificial Intelligence and Robotics** at the University of Hertfordshire (April 2025).

**Title:** *Reinforcement Learning for Robot Dexterous In-Hand Manipulation of Objects (Shadow Hand)*  
**Author:** Bharadwaj Rachuri  
**Supervisor:** Dr. Diego Resende Faria  
**Module:** 7COM1039-0206-2024 — Advanced Computer Science Masters Project
**Journal:** IJRES — Vol. 13, Issue 6, pp. 164–183 | ISSN: 2320-9364 | Impact Factor: 7.52 | [www.ijres.org](https://www.ijres.org)
---

## 🔮 Future Work

- **Domain randomisation** to improve policy generalisation across object shapes and sizes
- **Sim-to-real transfer** experiments on the physical Shadow Hand
- **Curriculum learning** to train progressively harder manipulation tasks
- **Multi-object generalisation** beyond pen rotation

---

## 👤 Author

**Bharadwaj Rachuri**  
MSc Artificial Intelligence and Robotics, University of Hertfordshire  
📧 bharadwaj.r.career@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/bharadwajrachuri) | [GitHub](https://github.com/br23aay)

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
