#!/usr/bin/python
import gym
import sys
import os
from gym import wrappers
from pyvirtualdisplay import Display
import cv2
import time
import numpy as np
import tensorflow as tf

def prepare_data(x):
    x = x / 255
    wh = 16
    mask = cv2.inRange(x, np.array([0, 0., 0]), np.array([1, 0.5, 1])) / 255
    x = cv2.resize(x[84:, :, :], (wh, wh))
    mask = cv2.resize(mask, (wh, wh)).reshape([wh, wh, 1])
    #x += mask
    cv2.imshow('mask', x)
    cv2.waitKey(1)
    x = np.concatenate([x, mask], axis=2)
    return x - 0.5

GAME = 'CarRacing-v0'
MAX_STEPS = 200
MAX_EPOCHS = 1000
DECAY_FACTOR = 0.8
possible_actions = np.array([
    [-1., 0.0, 0.0], # left
    [1.0, 0.0, 0.0], # right
    [0.0, 1.0, 0.0], # go
    [0.0, 0.0, 0.8], # break
    [0.0, 0.0, 0.0], # rest
]).astype(np.float32)
env = gym.make(GAME)

in_dimen = (16, 16, 4)
out_dimen = (3,)

# Model
in_s = tf.placeholder(tf.float32, in_dimen)
in_s_next = tf.placeholder(tf.float32, in_dimen)
in_action = tf.placeholder(tf.int32, [])
in_reward = tf.placeholder(tf.float32, [])
def _q_function(s, a=None, dropout=False):
    _hidden = tf.layers.dense(s, 20, activation=tf.nn.relu, name='dense_1')
    if dropout:
        _hidden = tf.nn.dropout(_hidden, 0.5)
    _hidden = tf.layers.dense(_hidden, 30, activation=tf.nn.relu, name='dense_2')
    if dropout:
        _hidden = tf.nn.dropout(_hidden, 0.5)
    _all_q = tf.layers.dense(_hidden, len(possible_actions), name='dense_3')
    if a is not None:
        _Q =  _all_q[:, a]
    else:
        _Q = tf.reduce_max(_all_q, 1)
    return _Q, _all_q
q_function = tf.make_template('q_function', _q_function)
_s = tf.reshape(in_s, [1, np.prod(in_dimen)])
_s_next = tf.reshape(in_s_next, [1, np.prod(in_dimen)])
Q, all_q = q_function(_s, in_action, dropout=True)
Q_next, _ = q_function(_s_next)
new_Q = in_reward + DECAY_FACTOR * Q_next
loss = tf.reduce_sum((Q - tf.stop_gradient(new_Q))**2)

# Tensorflow operations
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(init)
saver.restore(sess, "./tmp/model.ckpt")

display = Display(visible=0, size=(1400, 900))
display.start()

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
        state = env.reset()
        score_round = 0
        for j in range(1000):
            tick = counter + 1
            counter = tick
            action = all_q.eval({in_s: prepare_data(state)})
            action = [0, 1, 0]
            state, reward, done, info = env.step(action)
            env.render()
            print(j)
            if done and env.env.tile_visited_count == len(env.env.track):
                total_score += 1000 - 0.1 * j
                print("Episode {0} finished".format(i + 1), 1000 - 0.1 * j)
                # env.monitor.close()
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
