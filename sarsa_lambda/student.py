# This code calculates the mean of the array and prints it to the screen

import numpy as np
import random

from tqdm import tqdm


def epsilon_greedy_action(env, Q, state, epsilon):
    if np.random.random() < epsilon:
        # choose a random action: exploration
        action = env.action_space.sample()
    else:
        # choose the action with the highest Q value for the current state
        # exploitation
        action = np.argmax(Q[state, :])
    return action


def sarsa_lambda(env, alpha=0.2, gamma=0.99, lambda_= 0.9, initial_epsilon=1.0, n_episodes=10000 ):

    ####### Hyperparameters
    # alpha = learning rate
    # gamma = discount factor
    # lambda_ = elegibility trace decay
    # initial_epsilon = initial epsilon value
    # n_episodes = number of episodes

    ############# define Q table and initialize to zero
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    E = np.zeros((env.observation_space.n, env.action_space.n))
    print("TRAINING STARTED")
    print("...")
    # init epsilon
    epsilon = initial_epsilon

    received_first_reward = False

    # Initialize the environment
    state = env.reset()
    state = torch.from_numpy(state).float()

    for ep in tqdm(range(n_episodes)):
        ep_len = 0
        state, _ = env.reset()
        action = epsilon_greedy_action(env, Q, state, epsilon)
        done = False
        while not done:
            ############## simulate the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_len += 1
            # env.render()
            next_action = epsilon_greedy_action(env, Q, next_state, epsilon)

            # TODO update q table and eligibility
            #calculate the TD error : reward + gamma * Q[next_state, next_action] - Q[state, action]
            td_error = reward + gamma * Q[next_state, next_action] - Q[state, action]
            
            #update the eligibility trace
            E[state, action] += 1
            
            #for each state and action update the Q table
            for s in range(env.observation_space.n):
                for a in range(env.action_space.n):
                    Q[s, a] += alpha * td_error * E[s, a]
                    E[s, a] = gamma * lambda_ * E[s, a]
            
            #check if the first reward is received    
            if not received_first_reward and reward > 0:
                received_first_reward = True
                print("Received first reward at episode ", ep)
            # update current state
            state = next_state
            action = next_action
        
        # print(f"Episode {ep} finished after {ep_len} steps.")

        # update current epsilon
        if received_first_reward:
            epsilon = 0.99 * epsilon
    print("TRAINING FINISHED")
    return Q

   # Initializing the environment
    state = env.reset()
    state = torch.from_numpy(state).float()

    for ep in tqdm(range(n_episodes)):
        ep_len = 0
        state, _ = env.reset()
        action = epsilon_greedy_action(env, Q, state, epsilon)
        done = False
        while not done:
            ############## simulate the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_len += 1
            # env.render()
            next_action = epsilon_greedy_action(env, Q, next_state, epsilon)

            # TODO update q table and eligibility
            #calculate the TD error : reward + gamma * Q[next_state, next_action] - Q[state, action]
            td_error = reward + gamma * Q[next_state, next_action] - Q[state, action]
            
            #update the eligibility trace
            E[state, action] += 1
            
            #for each state and action update the Q table
            for s in range(env.observation_space.n):
                for a in range(env.action_space.n):
                    Q[s, a] += alpha * td_error * E[s, a]
                    E[s, a] = gamma * lambda_ * E[s, a]
            
            #check if the first reward is received    
            if not received_first_reward and reward > 0:
                received_first_reward = True
                print("Received first reward at episode ", ep)
            # update current state
            state = next_state
            action = next_action
        
        # print(f"Episode {ep} finished after {ep_len} steps.")

        # update current epsilon
        if received_first_reward:
            epsilon = 0.99 * epsilon
    print("TRAINING FINISHED")
    return Q