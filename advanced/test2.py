import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent2 import Agent
import cv2


env = gym.make('CarRacing-v0')
env.seed(2)
agent = Agent(state_size=32, action_size=3, random_seed=2, learn_every=32)


def ddpg(n_episodes=1000, max_t=1000, print_every=1, img_size=96):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        #state = cv2.resize(state, (img_size, img_size))
        agent.reset()
        score = 0
        for t in range(max_t):
            print('epoch = {} Step = {} Score = {}'.format(i_episode, t, score))
            action = agent.act(state.copy()[None])
            next_state, reward, done, _ = env.step(action[0])
            cv2.imwrite(f'test_{t}.jpg', next_state)
            #next_state = cv2.resize(next_state, (img_size, img_size))

            cv2.imwrite(f'test2_{t}.jpg', next_state)

            agent.step(state.copy()[None], action, reward, next_state[None], done)
            state = next_state
            score += reward

            env.render()
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores


scores = ddpg()