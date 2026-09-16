#RL FrozenLake (Q-Learning)
A simple Reinforcement Learning project where an agent learns to solve the **FrozenLake** game using **Q-learning** (tabular RL) with an **epsilon-greedy** exploration strategy.

## Demo / Goal
- Start at **S**
- Reach the **G** goal tile
- Avoid holes **H**
- Learn the best path by training over many episodes

This project is great for learning the basics of:
- States, actions, rewards
- Q-table
- Exploration vs exploitation
- Q-learning update rule
  
---

## Requirements
- Python 3.8+ (works with newer versions too)
- Packages:
     - `gymnasium`
     - `numpy`
     - (optional) `pygame` for rendering

Install:
   bash
pip install gymnasium numpy

How it works:
Environment
. FrozenLake-v1(4*4grid)
. Actions:
  - 0 = Left
  - 1 = Down
  - 2 = Right
  - 3 = Up
We update the Q-table using:
 Q(s,a)<-Q(s,a)+alpha[r+gammaMax Q(s',a')-Q(s,a)]
where alpha: learning rate
      gamma: discount factor
    
