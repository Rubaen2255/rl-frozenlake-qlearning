import numpy as np 
import gymnasium as gym

#------------- 1) Create environment -------------

env = gym.make("FrozenLake-v1", is_slippery=True)

n_states = env.observation_space.n 
n_actions = env.action_space.n

#------------- 2) Initialize Q-table -------------

Q = np.zeros((n_states, n_actions))

#------------- 3) Hyperparameters -------------

alpha = 0.8        
gamma = 0.95        
epsilon = 1.0      
epsilon_min = 0.05 
epsilon_decay = 0.999


episodes = 20000
max_steps = 100

#------------- 4) Helper: epsilon-greedy action -------------

def choose_action(state): 
    if np.random.rand() < epsilon: 
        return env.action_space.sample()      
    return int(np.argmax(Q[state]))          

#------------- 5) Training loop -------------

for ep in range(episodes): 
    state, info = env.reset() 
    done = False

for step in range(max_steps):
    action = choose_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Q-learning update rule:
    # Q[s,a] <- Q[s,a] + alpha * (reward + gamma*max(Q[s',:]) - Q[s,a])
    best_next = np.max(Q[next_state])
    td_target = reward + gamma * best_next
    td_error = td_target - Q[state, action]
    Q[state, action] += alpha * td_error

    state = next_state
    if done:
        break

# Decay exploration
epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Training finished.") 
print("Q-table:\n", Q)

#------------- 6) Test (run greedy policy) -------------

test_env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
state , info = test_env.reset()

done = False 
steps = 0 
while not done and steps < 50: 
    action = int(np.argmax(Q[state])) 
    state, reward, terminated, truncated, info = test_env.step(action) 
    done = terminated or truncated 
    steps += 1

print("Test finished. Reward:", reward) 
test_env.close()