import numpy as np
import pandas as pd

# Define states (using standardized or preprocessed data from earlier)
states = df[['Customer_Age', 'Months_on_book', 'Total_Trans_Amt_Log', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']].values

# Define action space: email, push notification, wait
actions = ['email', 'push', 'wait']

# Initialize Q-Table (number of states x number of actions)
q_table = np.zeros((len(states), len(actions)))

# Define hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration-exploitation trade-off

# Simulate a simple Q-learning update
for episode in range(1000):
    # Randomly choose an initial state
    state_idx = np.random.randint(0, len(states))
    state = states[state_idx]

    # Choose action based on epsilon-greedy
    if np.random.uniform(0, 1) < epsilon:
        action_idx = np.random.randint(0, len(actions))  # Explore
    else:
        action_idx = np.argmax(q_table[state_idx])  # Exploit

    # Assume we take action and get a reward from the environment (based on simulated interaction)
    reward = simulate_environment(state, actions[action_idx])

    # Update Q-value for the state-action pair
    next_state_idx = np.random.randint(0, len(states))  # Randomly simulate the next state
    q_table[state_idx, action_idx] = q_table[state_idx, action_idx] + alpha * (
        reward + gamma * np.max(q_table[next_state_idx]) - q_table[state_idx, action_idx]
    )

#visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Function to visualize the Q-table
def plot_q_table(q_table, actions, title="Q-Table Heatmap"):
    plt.figure(figsize=(12, 8))
    sns.heatmap(q_table, annot=True, cmap="viridis", xticklabels=actions, yticklabels=range(len(q_table)))
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.title(title)
    plt.show()

# Visualize the initial Q-table
plot_q_table(q_table, actions, title="Q-Table Heatmap (Before Learning)")

# Run the Q-learning algorithm (continued from above)
for episode in range(1000, 2000):
    # Randomly choose an initial state
    state_idx = np.random.randint(0, len(states))
    state = states[state_idx]

    # Choose action based on epsilon-greedy
    if np.random.uniform(0, 1) < epsilon:
        action_idx = np.random.randint(0, len(actions))  # Explore
    else:
        action_idx = np.argmax(q_table[state_idx])  # Exploit

    # Assume we take action and get a reward from the environment (based on simulated interaction)
    reward = simulate_environment(state, actions[action_idx])

    # Update Q-value for the state-action pair
    next_state_idx = np.random.randint(0, len(states))  # Randomly simulate the next state
    q_table[state_idx, action_idx] = q_table[state_idx, action_idx] + alpha * (
        reward + gamma * np.max(q_table[next_state_idx]) - q_table[state_idx, action_idx]
    )

# Visualize the Q-table after learning
plot_q_table(q_table, actions, title="Q-Table Heatmap (After Learning)")

# Optional: Create a function to simulate and plot decision-making for a specific customer
def simulate_decision_making(q_table, actions, customer_idx):
    state = states[customer_idx]
    state_idx = customer_idx
    best_action_idx = np.argmax(q_table[state_idx])
    best_action = actions[best_action_idx]
    
    print(f"Recommended action for customer {customer_idx}: {best_action}")

    # Visualize the Q-values for the selected state
    plt.figure(figsize=(10, 6))
    sns.barplot(x=actions, y=q_table[state_idx], palette="viridis")
    plt.title(f"Q-Values for Customer {customer_idx} (State {state_idx})")
    plt.xlabel("Actions")
    plt.ylabel("Q-Value")
    plt.show()

# Test the decision-making visualization for a specific customer (e.g., customer 0)
simulate_decision_making(q_table, actions, customer_idx=0)
