import time, math, random, bisect, copy
import gym
import numpy as np
import cv2
import tensorflow as tf

#tf.logging.set_verbosity(tf.logging.INFO)

def replayBestBots(bestNeuralNets, steps, sleep):  
    choice = input("Do you want to watch the replay ?[Y/N] : ")
    if choice=='Y' or choice=='y':
        for i in range(len(bestNeuralNets)):
            if (i+1)%steps == 0 :
                observation = env.reset()
                totalReward = 0
                for step in range(MAX_STEPS):
                    env.render()
                    time.sleep(sleep)
                    action = bestNeuralNets[i].getOutput(observation)
                    observation, reward, done, info = env.step(action)
                    totalReward += reward
                    if done:
                        observation = env.reset()
                        break
                print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %4d" % (i+1, bestNeuralNets[i].fitness, totalReward))


def recordBestBots(bestNeuralNets):  
    print("\n Recording Best Bots ")
    print("---------------------")
    env.monitor.start('Artificial Intelligence/'+GAME, force=True)
    observation = env.reset()
    for i in range(len(bestNeuralNets)):
        totalReward = 0
        for step in range(MAX_STEPS):
            env.render()
            action = bestNeuralNets[i].getOutput(observation)
            observation, reward, done, info = env.step(action)
            totalReward += reward
            if done:
                observation = env.reset()
                break
        print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %4d" % (i+1, bestNeuralNets[i].fitness, totalReward))
    env.monitor.close()


def uploadSimulation():
    API_KEY = open('/home/dollarakshay/Documents/API Keys/Open AI Key.txt', 'r').read().rstrip()
    gym.upload('Artificial Intelligence/'+GAME, api_key=API_KEY)

def prepare_data(x):
    x = x / 255
    x = cv2.resize(x, (16, 16))
    return x - 0.5

GAME = 'CarRacing-v0'
MAX_STEPS = 200
MAX_EPOCHS = 1000
DECAY_FACTOR = 0.95
possible_actions = np.array([
    [-1., 0.0, 0.0], # left
    [1.0, 0.0, 0.0], # right
    [0.0, 1.0, 0.0], # go
    [0.0, 0.0, 0.8], # break
    [0.0, 0.0, 0.0], # rest
]).astype(np.float32)
env = gym.make(GAME)
observation = env.reset()

in_dimen = prepare_data(observation).shape
out_dimen = env.action_space.shape

# Model
in_s = tf.placeholder(tf.float32, in_dimen)
in_s_next = tf.placeholder(tf.float32, in_dimen)
in_action = tf.placeholder(tf.int32, [])
in_reward = tf.placeholder(tf.float32, [])
def _q_function(s, a=None, dropout=False):
    _hidden = tf.layers.dense(s, 10, activation=tf.nn.relu, name='dense_1')
    if dropout:
        _hidden = tf.nn.dropout(_hidden, 0.5)
    _all_q = tf.layers.dense(_hidden, len(possible_actions), name='dense_2')
    if a is not None:
        _Q =  _all_q[:, a]
    else:
        _Q = tf.reduce_max(_all_q, 1)
    return _Q, _all_q
q_function = tf.make_template('q_function', _q_function)
_s = tf.reshape(in_s, [1, np.prod(in_dimen)])
_s_next = tf.reshape(in_s_next, [1, np.prod(in_dimen)])
Q, all_Q = q_function(_s, in_action, dropout=False)
Q_next, _ = q_function(_s_next)
new_Q = in_reward + DECAY_FACTOR * Q_next
loss = tf.reduce_sum((Q - tf.stop_gradient(new_Q))**2)

# Tensorflow operations
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

env.render()

runningReward = 0
train_samples = []
for epoch_i in range(MAX_EPOCHS):
    observation = env.reset()
    observation_prev = observation
    totalReward = 0
    temp_samples = []
    rewards = []
    for step in range(MAX_STEPS + 10 * (epoch_i + 1)):
        #if step % 10 == 0:
        env.render()
        #    pass
        prob = all_Q.eval({in_s: prepare_data(observation_prev)})
        if step % 10 == 0:
            print(prob)
        if random.random() < 0.9:
            action = np.argmax(prob)
        else:
            action = np.random.choice(len(possible_actions))
        #real_action = np.sum(possible_actions * prob.reshape([len(possible_actions), 1]), axis=0)
        real_action = possible_actions[action]# * prob[action]
        observation, reward, done, info = env.step(real_action)
        temp_samples.append({in_s: prepare_data(observation_prev), in_action: action,
                    in_s_next: prepare_data(observation)})
        rewards.append(reward)
        observation_prev = observation
        totalReward += reward
        if done:
            break
    for i in range(len(rewards)):
        mult = 1
        reward = 0
        for j in range(len(rewards) - i):
            reward += rewards[i + j] * mult
            mult *= DECAY_FACTOR
        temp_samples[i][in_reward] = rewards[i]
        train_samples.append(temp_samples[i])
    for _ in range(10):
        random.shuffle(train_samples)
        for ts in train_samples[:1000]:
            train_op.run(ts)
    train_samples = train_samples[:10000]
    print("Epoch : %3d  |  Reward : %5.0f  " % (epoch_i+1, totalReward) )

recordBestBots(bestNeuralNets)

uploadSimulation()

replayBestBots(bestNeuralNets, max(1, int(math.ceil(MAX_GENERATIONS/10.0))), 0.0625)

