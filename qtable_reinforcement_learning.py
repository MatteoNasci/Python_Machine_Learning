import numpy as np
import gym

env = gym.make('CliffWalking-v0')

print('Observation space: ', env.observation_space)
print('Action space: ', env.action_space)

Q_table_shape = [env.observation_space.n, env.action_space.n]
Q = np.zeros(Q_table_shape)

episodes = 10_000
best_reward = -float('inf')
best_episode = -1
epsilon = 0.02  # probabilità di fare roba a caso
learning_rate = 0.01

debug_draw = False
debug_draw_interval = 1_000

for episode in range(episodes):
    state = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, done, info = env.step(action)

        prediction = Q[state, action]
        target = reward + np.max(Q[next_state])
        error = target - prediction
        Q[state, action] += learning_rate * error

        state = next_state

        total_reward += reward

        if episode % debug_draw_interval == 0 and debug_draw:
            env.render()
            print('Action: ', action)
            print('Next_state: ', next_state, ', reward: ', reward)

    if total_reward > best_reward:
        best_reward = total_reward
        best_episode = episode
    print(Q)
    print('Best overall reward: ', best_reward, ' at episode: ', best_episode,
          ', current episode ', episode, ', total reward = ', total_reward)
