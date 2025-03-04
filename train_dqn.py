import sys
import pandas as pd
import numpy as np
from collections import deque
import tensorflow as tf
import random

# Load dataset
def load_and_preprocess_dataset(file_path):
    dataset = pd.read_csv(file_path)
    dataset = dataset.apply(pd.to_numeric, errors='coerce')
    dataset.fillna(0, inplace=True)
    return dataset

# DQN Agent Class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
    
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    urban_file, highway_file, num_episodes, batch_size = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    
    urban_dataset = load_and_preprocess_dataset(urban_file).head(100)
    highway_dataset = load_and_preprocess_dataset(highway_file).head(100)
    
    state_size = 4
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    
    for e in range(num_episodes):
        sample_data = urban_dataset.sample(frac=0.5).values
        for row in sample_data:
            state = row[:state_size]
            action = agent.act(state)
            reward = row[state_size]
            next_state = state
            done = False
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        print(f"Episode {e+1}/{num_episodes} completed.")
