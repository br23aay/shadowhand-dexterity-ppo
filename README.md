# ShadowHand Dexterity PPO

Reinforcement learning for dexterous in-hand object manipulation using PPO on the Shadow Hand. Includes training code, evaluation tools, and sim-to-real considerations.

## Overview
This project implements a Proximal Policy Optimization (PPO) agent to train the Shadow Hand robot to manipulate a pen within a MuJoCo simulation. The task requires precise control, tactile feedback processing, and dexterous coordination.

## Features
- Custom MuJoCo environment: `PenSpinEnv`
- PPO algorithm using Stable-Baselines3
- Sensor-based feedback control
- Sim-to-real transfer strategies
- Evaluation and performance metrics
- Manual and automated manipulation modes

## Technologies
- MuJoCo
- Python
- Stable-Baselines3
- Shadow Hand model
- PyTorch

## License
This project is licensed under the MIT License.
