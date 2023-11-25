import random
import numpy as np
# import gymnasium as gym
# import time
# from gymnasium import spaces
# import os
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
import pickle



class VanillaFeatureEncoder:
    def __init__(self, env):
        self.env = env
        
    def encode(self, state):
        return state
    
    @property
    def size(self): 
        return self.env.observation_space.shape[0]

# This code initializes the RBF encoder using the environment
# passed into the constructor.
# It also contains a function to encode the state passed into the function,
# and a function to return the size of the feature vector generated by the encoder.

class RBFFeatureEncoder:
    def __init__(self, env): # modify
        self.env = env
        self.encoder = RBFSampler(gamma = 1, n_components=100)
        observation_examples = np.array([env.observation_space.sample() for x in range(100)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Initialize RBF samplers with different parameters
        self.rbf_space = [
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ]
        self.design_matrix = FeatureUnion(self.rbf_space)
        self.design_matrix.fit(self.scaler.transform(observation_examples))
        
    def encode(self, state):
        scaled = self.scaler.transform(state.reshape(1, -1))
        state_features = self.design_matrix.transform(scaled)
        return state_features

        
    @property
    def size(self): # modify
        # TODO return the number of features
        size = self.encoder.n_components*self.rbf_space.__len__()
        return size
    
class TDLambda_LVFA:
    def __init__(self, env, feature_encoder_cls=RBFFeatureEncoder, alpha=0.01, alpha_decay=1, 
                 gamma=0.9999, epsilon=0.3, epsilon_decay=0.995, final_epsilon=0.05, lambda_=0.9): # modify if you want (e.g. for forward view)
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.shape = (self.env.action_space.n, self.feature_encoder.size)
        self.weights = np.random.random(self.shape)*0.01 
        self.traces = np.zeros(self.shape) 
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.lambda_ = lambda_
        self.episode = []
        
    def Q(self, feats):
        feats = feats.reshape(-1, 1)
        return self.weights@feats
    
    def update_transition_backwards(self, s, a, s_prime, reward, done): # modify
        s_feats = self.feature_encoder.encode(s)
        s_prime_feats = self.feature_encoder.encode(s_prime) 
        # TODO  update the weights for the current state
        #calculate delta td error
        delta = reward + self.gamma*self.Q(s_prime_feats).max() - self.Q(s_feats)[a]
        #decay traces
        self.traces = self.gamma*self.lambda_*self.traces
        #accumulate traces
        self.traces[a] = self.traces[a] + s_feats
        #substitute traces
        #self.traces[a] = s_feats  
        #update weights
        self.weights[a] += self.alpha*delta*self.traces[a]
    
    
    
    def update_transition_forwards(self, s, action, s_prime, reward, done): # modify #to test
        s_feats = self.feature_encoder.encode(s)
        s_prime_feats = self.feature_encoder.encode(s_prime) 
        self.episode.append((s_feats, action, reward,))
        
        if done:
            T = len(self.episode)
            G = np.zeros(T)
            for t in reversed(range(T)):
                _,_, reward_t = self.episode[t]
                G[t] = reward_t + (self.gamma*G[t+1] if t+1 < T else 0)
                
            for t in range(T):
                s_feats_t,action_t = self.episode[t]
                delta = G[t] - self.Q(s_feats_t)[action_t]
                self.weights[action_t] += self.alpha*delta*s_feats_t
                
            self.episode = []
        
                    
    def update_alpha_epsilon(self): # do not touch
        self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)
        self.alpha = self.alpha*self.alpha_decay

        
    def policy(self, state): # do not touch
        state_feats = self.feature_encoder.encode(state)
        return self.Q(state_feats).argmax()
    
    def epsilon_greedy(self, state, epsilon=None): # do not touch
        if epsilon is None: epsilon = self.epsilon
        if random.random()<epsilon:
            return self.env.action_space.sample()
        return self.policy(state)
       
        
    def train(self, n_episodes=200, max_steps_per_episode=200): # do not touch
        print(f'ep | eval | epsilon | alpha')
        for episode in range(n_episodes):
            done = False
            s, _ = self.env.reset()
            self.traces = np.zeros(self.shape)
            for i in range(max_steps_per_episode):
                
                action = self.epsilon_greedy(s) #choose action using epsilon greedy policy
                s_prime, reward, done, _, _ = self.env.step(action) #take action
                self.update_transition_backwards(s, action, s_prime, reward, done) #update the weights and traces
                #self.update_transition_forwards(s, action, s_prime, reward, done)
                
                s = s_prime #update the state
                
                if done: break
                
            self.update_alpha_epsilon() #update alpha and epsilon

            if episode % 20 == 0:
                print(episode, self.evaluate(), self.epsilon, self.alpha)
                
    def evaluate(self, env=None, n_episodes=10, max_steps_per_episode=200): # do not touch
        if env is None:
            env = self.env
            
        rewards = []
        for episode in range(n_episodes):
            total_reward = 0
            done = False
            s, _ = env.reset()
            for i in range(max_steps_per_episode):
                action = self.policy(s)
                
                s_prime, reward, done, _, _ = env.step(action)
                
                total_reward += reward
                s = s_prime
                if done: break
            
            rewards.append(total_reward)
            
        return np.mean(rewards)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname):
        return pickle.load(open(fname,'rb'))
