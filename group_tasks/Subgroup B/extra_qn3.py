import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
class MarketingEnv:
    def __init__(self, customer_data):
        self.customer_data = customer_data.copy()
        self.time = 0  # Initialize time step
        self.max_time = 30  # Define the maximum number of time steps (e.g., days)
        self.state = None
        self.done = False

    def reset(self):
        self.time = 0
        self.done = False
        # Initialize the customer state randomly or based on certain criteria
        self.state = self.customer_data.sample(1).to_dict('records')[0]
        return self.state

    def step(self, action):
        """
        Takes an action and returns the next state, reward, and done flag.
        Action: 0 (Do nothing), 1 (Send communication)
        """
        reward = 0
        self.time += 1

        # Simulate customer response based on action and state
        if action == 1:
            # Assume a simple probability model for customer response
            response_prob = self._calculate_response_prob(self.state)
            if random.random() < response_prob:
                reward = 10  # Positive reward for successful engagement
            else:
                reward = -1  # Negative reward for unsuccessful attempt
        else:
            reward = 0  # No reward for inaction

        # Update customer state (e.g., increase fatigue if over-communicated)
        self.state = self._update_state(self.state, action)

        # Check if the episode is done
        if self.time >= self.max_time:
            self.done = True

        return self.state, reward, self.done

    def _calculate_response_prob(self, state):
        # Simplified response probability based on customer attributes
        base_prob = 0.05
        # Increase probability if customer is in a high-value segment
        if state['Segment'] in [2, 3]:
            base_prob += 0.1
        # Adjust based on external factors (e.g., economic indicators)
        base_prob += state.get('GDP_Growth', 0) * 0.05
        return min(base_prob, 1.0)

    def _update_state(self, state, action):
        # Update state variables (e.g., increase fatigue)
        if action == 1:
            state['Fatigue'] = state.get('Fatigue', 0) + 1
        else:
            state['Fatigue'] = max(state.get('Fatigue', 0) - 0.1, 0)
        return state
class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def choose_action(self, state):
        state_key = self._state_to_key(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        predict = self.q_table[state_key][action]
        target = reward + self.gamma * np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += self.alpha * (target - predict)

    def _state_to_key(self, state):
        # Convert state dictionary to a tuple key
        return tuple(sorted(state.items()))
# Initialize environment and agent
actions = [0, 1]  # 0: Do nothing, 1: Send communication
env = MarketingEnv(df2)
agent = QLearningAgent(actions)

num_episodes = 1000
rewards_per_episode = []

for episode in range(num_episodes):
    state = env.reset()
    total_rewards = 0

    while not env.done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_rewards += reward

    rewards_per_episode.append(total_rewards)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_rewards}")
plt.figure(figsize=(12, 6))
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode')
plt.show()
# Extract Q-values for a specific state
state_examples = random.sample(list(agent.q_table.keys()), 5)

for state_key in state_examples:
    q_values = agent.q_table[state_key]
    state_dict = dict(state_key)
    print(f"State: {state_dict}")
    print(f"Q-Values: {q_values}\n")
# Prepare data for heatmap
state_action_values = []
for state_key, q_values in agent.q_table.items():
    state_dict = dict(state_key)
    fatigue = state_dict.get('Fatigue', 0)
    segment = state_dict['Segment']
    preferred_action = np.argmax(q_values)
    state_action_values.append({'Fatigue': fatigue, 'Segment': segment, 'Action': preferred_action})

heatmap_data = pd.DataFrame(state_action_values)

# Pivot the data
heatmap_pivot = heatmap_data.pivot_table(index='Segment', columns='Fatigue', values='Action', aggfunc='mean')

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_pivot, annot=True, fmt=".1f", cmap='coolwarm')
plt.title('Preferred Action Heatmap')
plt.xlabel('Fatigue Level')
plt.ylabel('Customer Segment')
plt.show()

