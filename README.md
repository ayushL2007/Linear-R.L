# Model-Based Reinforcement Learning for Inverted Pendulum

A complete implementation of a **Model-Based Reinforcement Learning (MBRL)** framework for stabilizing an inverted pendulum. This project combines classical control-inspired dynamics modeling with reinforcement learning and supervised learning approaches for next-state prediction.

The repository contains:

- A custom **Linear Reinforcement Learning Controller** for pendulum stabilization.
- A **Generalized Reinforcement Learning Dynamics Model** for learning environment transitions.
- A **Linear Regression Next-State Predictor** trained on simulated transitions.
- Synthetic data generation through simulation.
- Comparative analysis between learned and analytical dynamics models.

---

## Overview

The inverted pendulum is a classic control problem where the objective is to keep a pendulum balanced upright despite its naturally unstable dynamics.

Unlike purely model-free approaches, this project explicitly learns the system dynamics and uses them to improve control performance and sample efficiency.

The workflow consists of:

1. Generating state-transition data using a simulator.
2. Training supervised and RL-based dynamics models.
3. Learning a control policy for stabilization.
4. Comparing linear and generalized dynamics representations.
5. Evaluating long-term balancing performance.

---

## Features

### Model-Based Linear RL Controller

- Implemented entirely from scratch.
- Uses a learned transition model to estimate future states.
- Maintains pendulum stability within approximately **±3% angular deviation** from the target state.
- Demonstrates efficient stabilization with minimal control effort.

### Generalized RL Dynamics Model

A reinforcement-learning-based environment model capable of learning transition dynamics directly from interaction data.

**Key Characteristics**

- Captures nonlinear system behavior.
- Learns state transitions without relying on analytical equations.
- Generalizes across unseen state-action combinations.
- Can be used for planning and policy improvement.

### Linear Regression Next-State Predictor

A lightweight supervised dynamics model trained on simulator-generated transitions.

**Input**

```python
[angle, angular_velocity, cart_posn, cart_velocity, action]
```

**Output**

```python
[next_angle, next_angular_velocity, next_cart_posn, next_cart_velocity]
```

**Highlights**

- Fast training and inference.
- Interpretable dynamics approximation.
- Useful baseline for comparison against learned RL models.

---

## State Representation

The environment state is represented as:

```math
s = [θ, θ̇]
```

where:

- `θ` = Pendulum angle
- `θ̇` = Angular velocity

The action space consists of the applied torque:

```math
a = τ
```

---

## Reward Function

The reward function encourages upright balancing while penalizing excessive control effort.

```math
R = -(θ² + λθ̇² + βτ²)
```

This incentivizes:

- Maintaining the upright position.
- Minimizing oscillations.
- Reducing unnecessary torque.

---

## Dynamics Modeling

### Linear Regression Model

The regression model learns:

```math
s(t+1) = f(s(t), a(t))
```

using synthetic trajectories generated from the simulator.

### Generalized RL Model

The RL dynamics model learns transition behavior in the form:

```math
P(s(t+1) | s(t), a(t))
```

allowing future-state estimation and model-based planning.

Compared to linear regression, it is able to represent more complex nonlinear dynamics present in the pendulum system.

---

## Results

| Model | Task | Performance |
|---------|---------|---------|
| Linear Regression | Next-State Prediction | High prediction accuracy |
| Generalized RL Model | Dynamics Learning | Captures nonlinear transitions |
| Linear RL Controller | Stabilization | ±3% target-angle error |

### Observations

- The linear regression model provides fast and interpretable state predictions.
- The generalized RL model better represents nonlinear dynamics.
- The model-based RL controller successfully stabilizes the pendulum using learned transition information.
- Synthetic-data training proved sufficient for learning effective dynamics representations.

---

## Project Structure

```text
.
├── NS-Model/
│   ├── ns_pred.py
│
├── R.L Model/
│   ├── fitted_vi_rl.py
│   ├── reward.py
│
├── RL_InvertedPendulum.ipynb
├── inverted_pendulum_state.ipynb
│
└── README.md
```

---

## Tech Stack

- Python
- NumPy
- Matplotlib
- Reinforcement Learning
- Linear Regression
- Scientific Computing
- Control Systems

---

## Key Takeaway

This project demonstrates how **Model-Based Reinforcement Learning**, **supervised dynamics prediction**, and **reinforcement-learning-based environment modeling** can be combined to solve a classical control problem efficiently.

By comparing a **Linear Regression Next-State Predictor** with a **Generalized RL Dynamics Model**, the project highlights the tradeoff between interpretability and representational power while achieving reliable inverted-pendulum stabilization.
