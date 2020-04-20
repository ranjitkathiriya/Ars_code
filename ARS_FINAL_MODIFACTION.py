import os
import sys
import numpy as np
import gym
from gym import wrappers
import pybullet_envs
import seaborn as sns
from numpy import mean, absolute 
import numpy as np 
import matplotlib.pyplot as plt



class ARS_Hyper_params():
    
    def __init__(self,num_moves,num_action,env_name):
        self.epoch = 1
        self.episode_length = 1000
        self.step_size = 0.015
        self.no_directions = 60
        self.nb_best_directions = 20
        assert self.nb_best_directions <= self.no_directions, "b must be <= n_directions"
        self.noise = 0.025
        self.seeds = 1
        envirnment_name = env_name
        
        self.n = np.zeros(num_moves)
        self.mean = np.zeros(num_moves)
        self.mean_diff = np.zeros(num_moves)
        self.var = np.zeros(num_moves)
        
        self.theta = np.zeros((num_action, num_moves))
        
    def normalization_data(self,num_moves):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (num_moves - self.mean) / self.n
        self.mean_diff += (num_moves - last_mean) * (num_moves - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (num_moves - obs_mean) / obs_std
    
    def policy(self, inputs, delta = None, direction = None):
        if direction == "negative":  return (self.theta - self.noise*delta).dot(inputs)
        elif direction == "positive": return (self.theta + self.noise*delta).dot(inputs)
        elif direction == "other": return (self.theta * self.noise*delta).dot(inputs)
        else: return self.theta.dot(inputs)
        
    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.no_directions)]
    
    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += self.step_size / (self.nb_best_directions * sigma_r) * step
        
    def evaluations_d(self,ars,env, direction = None, delta = None):
        state = env.reset()
        done = False
        num_plays = 0.
        reward_sum = 0
        while not done and num_plays<self.episode_length:
            # ars.observe(state)
            state = ars.normalization_data(state)
            action = ars.policy(state, delta,direction)
            state, reward, done, _ = env.step(action)
            reward = max(min(reward, 1), -1)
            reward_sum += reward
            num_plays += 1
        return reward_sum

        
    def train(self, env, ars,graph_name):
        episodes =[]
        steps = []
        for episode in range(self.epoch):
            # init deltas and rewards
            deltas = ars.sample_deltas()
            reward_positive = [0]*self.no_directions
            reward_negative = [0]*self.no_directions
            reward_other = [0]*self.no_directions
    
            # positive directions
            for i in range(self.no_directions):
                reward_positive[i] = ars.evaluations_d(ars,env,"positive",deltas[i])
                reward_negative[i] = ars.evaluations_d(ars,env,"negative",deltas[i])
                reward_other[i] = ars.evaluations_d(ars,env,"other",deltas[i])
                
                # print("Pos",reward_positive)
                # print("Neg",reward_negative)
                # print("OTh",reward_other)
                
            
            # # negative directions
            # for i in range(self.no_directions):
            #     reward_other[i] = ars.evaluations_d(ars,env,"other",deltas[i])
            
            all_rewards = np.array(reward_negative + reward_positive + reward_other)
            print(all_rewards)
            sigma_r = mean(absolute(all_rewards - mean(all_rewards)))   # Change 1
    
            # sort rollouts wrt max(r_pos, r_neg) and take (hp.b) best
            scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(reward_positive,reward_negative))}
            print("Scores",scores)
            order = sorted(scores.keys(), key=lambda x:scores[x])[-self.nb_best_directions:]
            # print("order",order)
            rollouts = [(reward_positive[k], reward_negative[k], deltas[k]) for k in order[::-1]]
            # print("Score",rollouts)
    
            # update policy:
            ars.update(rollouts, sigma_r)
    
            # evaluate
            reward_evaluation = ars.evaluations_d(ars,env,"None")
         
            # finish, print:
            print('episode',episode,'reward_evaluation',reward_evaluation)
            episodes.append(episode)
            steps.append(reward_evaluation)
            
        ars.display_graph(steps,episodes,graph_name)
        
    def display_graph(self,steps,episodes,graph_name):
        sns.set(style='darkgrid')
        plt.figure(figsize=(15,8))
        sns_save = sns.lineplot(x=episodes, y=steps)
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.title("Half Cheeta Mean Absolute Deviation MinMax Norm.") # You can comment this line out if you don't need title
        fig = sns_save.get_figure()
        fig.savefig(graph_name) 
        
env_name = "HalfCheetahBulletEnv-v0" 
env = gym.make(env_name)
num_moves = env.observation_space.shape[0]
num_action = env.action_space.shape[0]
ars = ARS_Hyper_params(num_moves,num_action,env_name)
ars.train(env,ars,"cheeta_new.png")