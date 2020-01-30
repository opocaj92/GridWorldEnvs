# GridWorldEnvs
## Some GridWorld environments for OpenAI Gym

### Problem
GridWorld is a simple and famous benchmark problem in Reinforcement Learning. The environment presents a rectangular grid in which an agent, starting from a certain cell, has to reach another cell defined as a goal, observing only its actual position. The agent receives a certain reward when it reachs the goal while moving in the environment. There are a lot of variations to this problem, like having multiple goal cells, walls (cells that the agent cannot pass by), bombs (cells that give a big negative reward if stepped) or uncertainty in the actions' outcome.

Also multi-agent versions have been proposed, like the Pursers-Evaders problem, in which a set of pursuers (our agents) have to reach coordinately the location of the evaders in order to catch them, while also them are moving in the environment (in out environment they move at random, but they could also learn how to evade). The problem ends when all the evaders have been catched.

### Overview
This little environment for OpenAI Gym allows to learn these problems. The files ***gym_gridworld/envs/GridWorld.py*** and ***gym_gridworld/envs/PursuersEvaders.py*** represent the two different problems respectively. They can simply be used as any other OpenAI Gym environment with `env = gym.make("GridWorld-v0")` and `env = gym.make("PursuersEvaders-v0")`. Custom maps can be made using text file similar to the provided examples (files named ***map#.txt*** are for the GridWorld environment, while files named ***mmap#.txt*** are for the PursuersEvaders one).

The two files ***q_learning.py*** and ***multiagent_q_learning.py*** are two example solvers for these two environments using the Q-Learning algorithm.

### Author
*Castellini Jacopo*