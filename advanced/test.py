#!/usr/bin/python
import gym
import sys
import os
from gym import wrappers
from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1400, 900))
# display.start()

RENDER = False
try:
    test = 'CarRacing-v0'
    env = gym.make(test)
    env.seed(1442)

    env = wrappers.Monitor(env, '/tmp/{0}-1'.format(test[2:]), force=True)
    
    counter = 0
    total_score = 0
    for i in range(10):
        done = False
        env.reset()
        score_round = 0
        for j in range(1000):
            tick = counter + 1
            counter = tick
            action = env.action_space.sample() # take a random action
            action = [1, 1, 0]
            observation, reward, done, info = env.step(action)
            env.render()
            print(j)
            if done and j != 999:
                total_score += 1000-0.1*j
                print("Episode {0} finished".format(i + 1), 1000-0.1*j)
                #env.monitor.close()
                break
    print(total_score)
except KeyboardInterrupt:
    sys.exit()
except IndexError:
    print("Error! Missing Test.")
    print("Please run: ")
    print("{0} Test-v0".format(sys.argv[0]))
    sys.exit()
except gym.error.UnregisteredEnv:
    print("Test not found!")
    sys.exit()	
except gym.error.UnsupportedMode:
    print("This doesnt support render mode!")
