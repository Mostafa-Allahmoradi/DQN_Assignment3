# Reinforcement Learning Programming (CSCN8020) - Assignment 2

## 👥 Author

Mostafa Allahmoradi - 9087818

## 📖 Assignment Overview

This repository contains the implementation for Assignment 3 of the Reinforcement Learning Programming course. The project is divided into two primary components: training an agent to play Atari Pong using a Deep Q-Network (DQN), and exploring the exploration-exploitation trade-off through a Multi-Armed Bandit Casino Challenge.

## 🎮 Part 1: Deep Q-Learning (Atari Pong)
An Object-Oriented PyTorch implementation of the DQN algorithm designed to solve the PongDeterministic-v4 OpenAI Gym environment.

Key Features & Architecture:
* Observation Space: The environment's raw RGB frames are cropped, down-sampled, converted to grayscale, and normalized. To capture temporal dynamics (ball velocity and direction), the network takes a stack of 4 consecutive frames as input (84x80 resolution).
* Action Space: Uses the reduced set of 6 actions (NOOP, FIRE, Move right, Move left, Right Fire, Left Fire).
* Network: A Convolutional Neural Network (CNN) with 3 convolutional layers followed by 2 fully connected layers.
* Exploration: Utilizes an epsilon-greedy strategy, decaying from 1.0 to a minimum of 0.05 at a rate of 0.995 per episode.

Ablation Studies:
The project includes automated ablation tests to compare learning stability and efficiency across different hyperparameters:
1. Batch Size: Comparing the default 8 against 16.
2. Target Update Rate: Comparing target network updates every 10 episodes against every 3 episodes.

## 🎰 Part 2: Multi-Armed Bandit Casino Challenge (Exercise 1)
An interactive Jupyter Notebook demonstrating fundamental Reinforcement Learning concepts using a 10-arm slot machine simulation.

Key Concepts Explored:
* Stationary vs. Non-Stationary Environments: Adapting agents to environments where the true reward probabilities drift over time.
* The Epsilon Trade-off: Visualizing how different epsilon values affect an agent's ability to locate and exploit the optimal machine
* Step-Size Updates: Comparing Sample Average (1/N) updates for stationary environments against Constant-Alpha (Exponential Moving Average) updates for non-stationary environments.

## 📂 Repository Structure
* pong_dqn_oop.py: The core Object-Oriented PyTorch implementation containing the PongTrainer, DQNAgent, and ReplayBuffer classes.

* assignment3_utils.py: Helper utilities for cropping and preprocessing Atari frames.

* Pong_DQN_Assignment.ipynb: The primary notebook for running the Pong ablation studies, plotting the metrics (score per episode and 5-episode average), and rendering the trained agent.

* MultiArmedBandit_Workshop.ipynb: The notebook containing the OOP Casino Challenge simulation and technical reflections.

* requirements.txt: Pinned dependencies for environment replication.

* logs folder which contains the training log file

* outputs folder which holds the round 1 and round 2 submissions of MultiArmedBandit_Workshop.ipynb as well as graph images of Pong_DQN_Assignment.ipynb Jupyter notebooks

## 🚀 Quick Start
1. **Clone this repository:**

   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. **Create a Virtual Environment**
* Windows:
    ```bash
   python -m venv .venv
   ```

* macOS / Linux:
```bash
   python3 -m venv venv
   ```

3. **Activate the Virtual Environment**
* On Windows (Command Prompt):
    ```bash
   .venv\Scripts\Activate
   ```

* On macOS / Linux:
    ```bash
   source venv/bin/activate
   ```

4. **Install Required Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. Run the Pong Training & Plots
Open Pong_DQN_Assignment.ipynb and run all cells. The script will automatically default to GPU acceleration if CUDA is available, train the models, save the ablation graphs, and generate a pong_model.pth weights file.

6. Watch the Agent Play
Once trained, the final cell in the Pong notebook will load the saved .pth weights and render a PyGame window so you can watch the agent compete against the built-in AI!