import time, math, random, bisect, copy
import gym
import numpy as np
import cv2
import tensorflow as tf

#tf.logging.set_verbosity(tf.logging.INFO)

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
DECAY_FACTOR = 0.95
possible_actions = np.array([
    [-1., 0.0, 0.0], # left
    [1.0, 0.0, 0.0], # right
    [0.0, 0.2, 0.0], # go
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
Q, all_q = q_function(_s, in_action, dropout=False)
Q_next, _ = q_function(_s_next)
new_Q = in_reward + DECAY_FACTOR * Q_next
loss = tf.reduce_sum((Q - tf.stop_gradient(new_Q))**2)

# Tensorflow operations
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(init)
saver.restore(sess, "./tmp/model.ckpt")

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
        prob = all_q.eval({in_s: prepare_data(observation_prev)})
        if step % 10 == 0:
            print(prob)
        if random.random() < 0.9:
            action = np.argmax(prob)
        else:
            action = np.random.choice(len(possible_actions))
        #real_action = np.sum(possible_actions * prob.reshape([len(possible_actions), 1]), axis=0)
        real_action = possible_actions[action]# * prob[action]
        observation, reward, done, info = env.step(real_action)
        reward = min(reward, 2)
        reward = max(reward, -2)
        #if reward < 0:
        #    reward = -10
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
    for _ in range(min(10, len(train_samples) // 100)):
        random.shuffle(train_samples)
        for ts in train_samples[:1000]:
            train_op.run(ts)
    train_samples = train_samples[:100000]
    print("Epoch : %3d  |  Reward : %5.0f  " % (epoch_i+1, totalReward) )
    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)

