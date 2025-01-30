# Reinforcement Learning Agents

This repository contains implementations of three Deep Reinforcement Learning (RL) algorithms:

- **DQN (Deep Q-Network)**
- **DDPG (Deep Deterministic Policy Gradient)**
- **SAC (Soft Actor-Critic)**

These algorithms are implemented to work with four different robotics tasks from **Gymnasium Robotics**.

---
## Installation

Before running the scripts, install the required dependencies:

```bash
pip install -r requirements.txt
```

---
## Running the Scripts

Each script can be executed from the command line. Below are the details for running each algorithm.

### **DQN (Deep Q-Network)**

Run DQN with default settings:
```bash
python DQN.py
```

### **DDPG (Deep Deterministic Policy Gradient)**

Run DDPG with default settings:
```bash
python DDPG.py
```

### **SAC (Soft Actor-Critic)**

Run SAC with default settings:
```bash
python SAC.py
```

---
## Available Tasks

These scripts support four different robotics tasks from **Gymnasium Robotics**:

1. **FetchReach-v4**  - A simple reaching task where the robot arm must reach a target position (DQN, DDPG, SAC).
2. **FetchPush-v4**  - The robot must push an object to a target location (DDPG, SAC).
3. **FetchSlide-v4** - The robot must slide an object to a target location (DDPG, SAC).
4. **FetchPickAndPlace-v4** - The robot must pick up and place an object at a target location (DDPG, SAC).

To change the environment, you must manually edit the respective Python script (`DQN.py`, `DDPG.py`, or `SAC.py`) and set the `env_name` variable to the desired environment.

---
## Hyperparameters

Each script has predefined hyperparameters that need to be manually adjusted within the script files (`DQN.py`, `DDPG.py`, or `SAC.py`). Below are some key hyperparameters:

| Parameter | Description | Default Value |
|----------|-------------|---------------|
| `env_name` | Environment name | `FetchReach-v4` |
| `episodes` | Number of training episodes | `500` |
| `learning_rate` | Learning rate for optimizer | `0.001` (DQN), `0.0003` (DDPG, SAC) |
| `gamma` | Discount factor for rewards | `0.99` |
| `batch_size` | Number of samples per training batch | `64` |
| `tau` | Soft update parameter for target networks | `0.005` |
| `hidden_size` | Number of neurons per hidden layer | `256` |

To customize these hyperparameters, open the respective script and modify the values accordingly. 

---
## Notes
- Ensure that `gymnasium` and `mujoco` dependencies are installed for running Gymnasium Robotics environments.
- Default hyperparameters are chosen based on common RL practices but can be tuned based on performance.
- Training time varies based on the environment and hyperparameters.

Happy Reinforcement Learning! 🚀

