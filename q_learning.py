import numpy as np
import gym
import gym_gridworld
import itertools
from collections import defaultdict

env = gym.make("GridWorld-v0")

def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.n_actions)
    env.reset()
    env.render()
    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes))
        observation = env.reset()
        for i in itertools.count():
            a_prob = policy(observation)
            a = np.random.choice([i for i in range(len(a_prob))], p = a_prob)
            next_observation, reward, done, _ = env.step(a)
            best_next_a = np.argmax(Q[next_observation])
            Q[observation][a] += alpha * (reward + discount_factor * Q[next_observation][best_next_a] - Q[observation][a])
            if done:
                break
            observation = next_observation
    return Q

Q = q_learning(env, 500)
print(Q)