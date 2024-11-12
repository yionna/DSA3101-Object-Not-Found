import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Assuming df2 is the preprocessed customer data as per previous instructions

# Define frequency and channel options
frequencies = ['Daily', 'Every 3 Days', 'Weekly', 'Bi-weekly']
channels = ['Email', 'SMS', 'Social Media', 'Push Notification']

# Create action space as a list of all possible combinations
actions = [(freq, chan) for freq in frequencies for chan in channels]
action_indices = {action: idx for idx, action in enumerate(actions)}

# Marketing Environment Class
class MarketingEnv:
    def __init__(self, customer_data):
        self.customer_data = customer_data.copy()
        self.max_time = 30  # Simulation over 30 days
        self.reset()

    def reset(self):
        self.time = 0
        self.done = False
        self.customer = self.customer_data.sample(1).to_dict('records')[0]
        self.state = {
            'Segment': self.customer['Segment'],
            'Preferred_Channel': self.customer.get('Preferred_Channel', 'Email'),
            'Last_Interaction_Days': np.inf,
            'Communications_Received': 0,
            'Clicks': 0,
            'GDP_Growth': self.customer.get('GDP_Growth', 0),
            'Competitor_Promotion': self.customer.get('Competitor_Promotion', 0),
        }
        return self.state

    def step(self, action_index):
        action = actions[action_index]
        frequency, channel = action
        reward = 0
        self.time += 1
        self.state['Last_Interaction_Days'] += 1

        send_communication = False
        if frequency == 'Daily':
            send_communication = True
        elif frequency == 'Every 3 Days' and self.time % 3 == 0:
            send_communication = True
        elif frequency == 'Weekly' and self.time % 7 == 0:
            send_communication = True
        elif frequency == 'Bi-weekly' and self.time % 14 == 0:
            send_communication = True

        if send_communication:
            self.state['Communications_Received'] += 1
            self.state['Last_Interaction_Days'] = 0
            click_prob = self._calculate_click_prob(channel)
            if random.random() < click_prob:
                reward = 10
                self.state['Clicks'] += 1
            else:
                reward = -1
            if self.state['Communications_Received'] > 5:
                reward -= 5

        if self.time >= self.max_time:
            self.done = True

        return self.state.copy(), reward, self.done

    def _calculate_click_prob(self, channel):
        base_ctr = 0.02
        if channel == self.state['Preferred_Channel']:
            base_ctr += 0.05
        if self.state['Segment'] in [2, 3]:
            base_ctr += 0.03
        base_ctr += self.state['GDP_Growth'] * 0.01
        base_ctr += self.state['Competitor_Promotion'] * -0.02
        base_ctr -= self.state['Communications_Received'] * 0.005
        return max(min(base_ctr, 1.0), 0.0)

# Q-Learning Agent Class
class QLearningAgent:
    def __init__(self, action_space_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.action_space_size = action_space_size

    def choose_action(self, state):
        state_key = self._state_to_key(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        predict = self.q_table[state_key][action]
        target = reward + self.gamma * np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += self.alpha * (target - predict)

    def _state_to_key(self, state):
        key_items = (
            state['Segment'],
            state['Preferred_Channel'],
            min(state['Last_Interaction_Days'], 7),
            min(state['Communications_Received'], 10),
            state['GDP_Growth'],
            state['Competitor_Promotion'],
        )
        return key_items

# Training the Agent
env = MarketingEnv(df2)
agent = QLearningAgent(action_space_size=len(actions))

num_episodes = 5000
rewards_per_episode = []

for episode in range(num_episodes):
    state = env.reset()
    total_rewards = 0

    while not env.done:
        action_index = agent.choose_action(state)
        next_state, reward, done = env.step(action_index)
        agent.learn(state, action_index, reward, next_state)
        state = next_state
        total_rewards += reward

    rewards_per_episode.append(total_rewards)

    if (episode + 1) % 500 == 0:
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_rewards}")

# Visualization of Total Rewards
plt.figure(figsize=(12, 6))
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode')
plt.show()

# Policy Analysis and Heatmap Visualization
policy = {}
for state_key, q_values in agent.q_table.items():
    best_action_index = np.argmax(q_values)
    best_action = actions[best_action_index]
    policy[state_key] = best_action

policy_df = pd.DataFrame([
    {
        'Segment': state_key[0],
        'Preferred_Channel': state_key[1],
        'Last_Interaction_Days': state_key[2],
        'Communications_Received': state_key[3],
        'GDP_Growth': state_key[4],
        'Competitor_Promotion': state_key[5],
        'Frequency': best_action[0],
        'Channel': best_action[1],
    }
    for state_key, best_action in policy.items()
])

# Frequency Heatmap
plt.figure(figsize=(12, 6))
sns.countplot(data=policy_df, x='Frequency', hue='Segment')
plt.title('Preferred Communication Frequency by Segment')
plt.xlabel('Frequency')
plt.ylabel('Count')
plt.legend(title='Segment')
plt.show()

# Channel Heatmap
plt.figure(figsize=(12, 6))
sns.countplot(data=policy_df, x='Channel', hue='Preferred_Channel')
plt.title('Preferred Channels vs. Customer Preferences')
plt.xlabel('Channel')
plt.ylabel('Count')
plt.legend(title='Preferred Channel')
plt.show()
