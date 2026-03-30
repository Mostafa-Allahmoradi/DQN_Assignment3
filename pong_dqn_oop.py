import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import time
import os
import logging
from datetime import datetime
from assignment3_utils import process_frame

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_shape, action_size, batch_size=8, target_update=10):
        self.state_shape = state_shape
        self.action_size = action_size
        self.device = DEVICE
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = batch_size
        self.target_update_rate = target_update
        
        self.policy_net = DQN(state_shape, action_size).to(self.device)
        self.target_net = DQN(state_shape, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayBuffer(10000)

    def act(self, state, is_playing=False):
        if not is_playing and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(np.array([state])).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_state).max(1)[0]
        expected_q_values = reward + (self.gamma * next_q_values * (1 - done))
        
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

class PongTrainer:
    def __init__(self, batch_size=8, target_update_rate=10):
        self.batch_size = batch_size
        self.target_update_rate = target_update_rate
        
        # Determine action size from a dummy environment
        dummy_env = gym.make("PongDeterministic-v4")
        action_size = dummy_env.action_space.n
        dummy_env.close()
        
        self.state_shape = (4, 84, 80)
        self.agent = DQNAgent(self.state_shape, action_size, batch_size, target_update_rate)
        
        # Initialize Logger
        self.logger = self._setup_logger()

    def _setup_logger(self):
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_b{self.batch_size}_t{self.target_update_rate}_{timestamp}.log")

        logger = logging.getLogger(f"PongTrainer_{self.batch_size}_{self.target_update_rate}")
        logger.setLevel(logging.INFO)

        # Prevent duplicate handlers if multiple instances are created
        if not logger.handlers:
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            
            formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            logger.addHandler(fh)
            logger.addHandler(ch)
            
        return logger

    def train(self, episodes=150):
        self.logger.info(f"Starting training loop: {episodes} episodes | Batch: {self.batch_size} | Target Update: {self.target_update_rate}")
        env = gym.make("PongDeterministic-v4")
        scores = []
        avg_last_5_scores = []
        step_history = []
        total_steps = 0
        
        for e in range(episodes):
            obs = env.reset()
            if isinstance(obs, tuple): obs = obs[0]
            processed_obs = process_frame(obs, (84, 80)).squeeze()
            state_stack = deque([processed_obs]*4, maxlen=4)
            current_state = np.array(state_stack)
            
            done = False
            score = 0
            
            while not done:
                action = self.agent.act(current_state)
                step_result = env.step(action)
                
                if len(step_result) == 5:
                    next_obs, reward, done, truncated, _ = step_result
                    done = done or truncated
                else:
                    next_obs, reward, done, _ = step_result
                    
                total_steps += 1
                processed_next = process_frame(next_obs, (84, 80)).squeeze()
                state_stack.append(processed_next)
                next_state = np.array(state_stack)
                
                self.agent.memory.push(current_state, action, reward, next_state, done)
                current_state = next_state
                score += reward
                self.agent.replay()
                
            self.agent.decay_epsilon()
            if e % self.agent.target_update_rate == 0:
                self.agent.update_target_network()
                
            scores.append(score)
            last_5_avg = np.mean(scores[-5:]) if len(scores) >= 5 else np.mean(scores)
            avg_last_5_scores.append(last_5_avg)
            step_history.append(total_steps)
            
            self.logger.info(f"Episode: {e+1}, Score: {score}, Avg(Last 5): {last_5_avg:.2f}, Epsilon: {self.agent.epsilon:.3f}")
            
        env.close()
        self.logger.info("Training complete.")
        return step_history, scores, avg_last_5_scores

    def play(self, episodes=3):
        self.logger.info("🎮 Launching game window. Watch the agent play!")
        try:
            env = gym.make("PongDeterministic-v4", render_mode="human")
        except:
            env = gym.make("PongDeterministic-v4")
            
        for e in range(episodes):
            obs = env.reset()
            if isinstance(obs, tuple): obs = obs[0]
            processed_obs = process_frame(obs, (84, 80)).squeeze()
            state_stack = deque([processed_obs]*4, maxlen=4)
            current_state = np.array(state_stack)
            
            done = False
            score = 0
            
            while not done:
                try: env.render() 
                except: env.render(mode="human")
                time.sleep(0.015)
                
                action = self.agent.act(current_state, is_playing=True)
                step_result = env.step(action)
                
                if len(step_result) == 5:
                    next_obs, reward, done, truncated, _ = step_result
                    done = done or truncated
                else:
                    next_obs, reward, done, _ = step_result
                    
                processed_next = process_frame(next_obs, (84, 80)).squeeze()
                state_stack.append(processed_next)
                current_state = np.array(state_stack)
                score += reward
                
            self.logger.info(f"Match {e+1} Final Score: {score}")
        env.close()

    def save(self, filename="pong_model.pth"):
        torch.save(self.agent.policy_net.state_dict(), filename)
        self.logger.info(f"💾 Model saved to {filename}")

    def load(self, filename="pong_model.pth"):
        if os.path.exists(filename):
            self.agent.policy_net.load_state_dict(torch.load(filename, map_location=self.agent.device))
            self.agent.policy_net.eval()
            self.logger.info(f"📂 Model loaded from {filename}")
        else:
            self.logger.error("❌ Error: File not found.")