# Ethical Decision-Making for Autonomous Driving using Deep Reinforcement Learning in CARLA

This project extends the work from [Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning](https://github.com/idreesshaikh/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning) with additional development focused on ethical decision-making scenarios for autonomous vehicles.

Developed as part of a final year Computer Science dissertation project at Swansea University.

---

## Overview

This project investigates the behaviour of reinforcement learning models when presented with ethical driving scenarios within the CARLA autonomous driving simulator.

The system implements an end-to-end autonomous driving solution using **Proximal Policy Optimization (PPO)** trained on feature representations extracted through a **Variational Autoencoder (VAE)**.

The project extends autonomous driving functionality with custom ethical scenarios inspired by research including:

- *The Moral Machine Experiment* (Awad et al., 2018)
- *Ethical Decision Making Behind the Wheel – A Driving Simulator Study* (Samuel et al., 2020)

The aim of the project is to evaluate how reinforcement learning agents behave when presented with moral and ethical dilemmas while driving.

---

## Features

- End-to-end autonomous driving in CARLA
- PPO reinforcement learning implementation
- Variational Autoencoder (VAE) state representation
- Custom ethical driving scenarios
- Pretrained model checkpoints included
- Scenario-based testing and evaluation
- CARLA simulator integration

---

## Tested Environment

The project was developed and tested using:

- Python 3.7+
- CARLA Simulator
- Unreal Engine 5
- PyTorch
- Windows 11

GPU acceleration is strongly recommended for training and evaluation.

---

## Setup

For detailed dependency installation and CARLA setup instructions, refer to the original repository:

https://github.com/idreesshaikh/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning

### Quick Setup

1. Clone this repository
2. Follow CARLA and dependency setup from the original repository
3. Install required Python packages
4. Launch the CARLA simulator
5. Run training or evaluation scripts

---

## Usage

### Test a Trained Agent

```bash
python continuous_driver.py --exp-name ppo --train False --town Town07 --scenario Scenario01
```

### Train an Agent

```bash
python continuous_driver.py --exp-name ppo --train True --town Town07 --scenario Scenario01
```

---

## Available Scenarios

### Scenario01 — Jaywalker Scenario

A pedestrian unexpectedly crosses the road, requiring the agent to react safely while maintaining vehicle control.

### Scenario02 — Ethical Dilemma Scenario

The agent is presented with a trolley-problem-inspired ethical scenario involving unavoidable collision outcomes, requiring the model to minimise harm.

These scenarios were inspired by existing research into ethical decision-making and autonomous vehicles.

---

## Command Options

| Option | Description |
|---|---|
| `--town` | CARLA town environment (Town07, Town02, etc.) |
| `--scenario` | Scenario configuration (Scenario01, Scenario02) |
| `--train` | `True` to train the model, `False` to evaluate pretrained models |

---

## Pretrained Models

The repository includes pretrained model checkpoints used during testing and evaluation within the dissertation project.

Models can be loaded directly through the evaluation command:

```bash
python continuous_driver.py --exp-name ppo --train False --town Town07 --scenario Scenario01
```

Checkpoint files are located within the `checkpoints/` directory.

> Note: Due to the stochastic nature of reinforcement learning, results may vary slightly between executions.

---

## Project Structure

| File / Directory | Description |
|---|---|
| `continuous_driver.py` | Main PPO training and evaluation script |
| `discrete_driver.py` | Experimental discrete action implementation |
| `autoencoder/` | Variational Autoencoder implementation |
| `networks/` | Reinforcement learning network architectures |
| `simulation/` | CARLA environment integration |
| `checkpoints/` | Pretrained model checkpoints |
| `parameters.py` | Hyperparameters and configuration |

---

## Dissertation Context

This repository accompanies the dissertation project:

> **Applying Machine Learning Technology to Ethical Scenarios using CARLA**

The project investigates:
- Reinforcement learning for autonomous driving
- Ethical decision-making in autonomous systems
- Scenario-based behavioural evaluation
- Comparison to prior human ethical decision-making studies

---

## Notes

- Training reinforcement learning agents in CARLA is computationally intensive.
- GPU acceleration is recommended.
- CARLA simulator assets and model checkpoints may require significant storage space.
- Results may vary slightly between executions due to stochastic training behaviour.

---

## Built With

- CARLA Simulator
- Unreal Engine 5
- PyTorch
- Python

---

## License

See `LICENSE.md` for licensing information.

---

## Acknowledgments

Original repository and baseline implementation by Idrees Razak:

https://github.com/idreesshaikh

Additional ethical scenario development and dissertation enhancements by Ryan Smith.
