import time, math, random, bisect, copy
import gym
import numpy as np
import cv2
import tensorflow as tf

#GAME = 'CarRacing-v0'
GAME = 'CartPole-v0'

MAX_STEPS = 200
MAX_EPOCHS = 1000
DECAY_FACTOR = 0.8
NOISE_DECAY = 0.8
env = gym.make(GAME)
observation = env.reset()

in_dimen = observation.shape
out_dimen = env.action_space.shape
print('in_dimen', in_dimen)
print('out_dimen', out_dimen)

# Model
# inputs
in_s = tf.placeholder(tf.float32, in_dimen)
in_s_prev = tf.placeholder(tf.float32, in_dimen)
in_action = tf.placeholder(tf.float32, out_dimen)
in_reward = tf.placeholder(tf.float32, [])
# hidden layers
x1 = tf.reshape(in_s, [1, np.prod(in_dimen)])
x2 = tf.reshape(in_s_prev, [1, np.prod(in_dimen)])
x = tf.concat([x1, x2], 1)
x = tf.layers.dense(x, 10, activation=tf.nn.relu, name='hidden_1')
#x = tf.layers.dense(x, 100, activation=tf.nn.relu, name='hidden_2')
#x = tf.nn.dropout(x, 0.5)
# normalization
x = x - tf.reduce_mean(x)
x = x / tf.sqrt(tf.reduce_mean(x**2))
a0 = tf.layers.dense(x, np.prod(out_dimen), name='a0')
'''
if GAME == 'CarRacing-v0':
    a0 = tf.nn.sigmoid(a0)
    a0 = a0 * np.array([[2., 1., 1.]])
    a0 = a0 + np.array([[-1., 0., 0.]])
'''
def std_activation(x):
    return 1e-8 + tf.exp(tf.clip_by_value(x, -10, 10))
    #return 1e-6 + (tf.nn.relu(x + 5) + tf.nn.sigmoid(x + 5)) * 1000
a_koef = tf.layers.dense(x, np.prod(out_dimen), activation=std_activation, name='a_koef')
a_koef1 = tf.layers.dense(x, np.prod(out_dimen), name='a_koef1')
mu_Q0 = tf.layers.dense(x, 1, name='a_const')
sigma_Q0 = tf.layers.dense(x, 1, activation=std_activation, name='sigma_Q')
parabola = tf.reduce_sum(a_koef * (in_action - a0)**2, 1, keepdims=True)
sigma_Q = sigma_Q0 + tf.reduce_sum(a_koef1 * (in_action - a0))
mu_Q = mu_Q0 - parabola
loss = tf.reshape((mu_Q - in_reward)**2 / (2 * sigma_Q**2 + 1e-8) + tf.log(sigma_Q + 1e-8) + 0.918938533, [])
sigma_a = sigma_Q0 / (2 * a_koef)
a_best = a0 #+ a_koef1 / (2 * a_koef)
a_random = tf.concat([tf.reshape(a_best, [1] + list(out_dimen)),
                      tf.reshape(sigma_a, [1] + list(out_dimen))], 0)

# Tensorflow operations
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
gvs = optimizer.compute_gradients(loss)
gvs = [(grad if grad is None else tf.clip_by_norm(grad, 0.2), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(gvs, tf.train.get_global_step())
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

env.render()

runningReward = 0
train_samples = []
for epoch_i in range(MAX_EPOCHS):
    observation = env.reset()
    observation_prev = observation
    total_reward = 0
    running_noise = np.zeros(out_dimen)
    temp_samples = []
    rewards = []
    for step in range(MAX_STEPS + 10 * (epoch_i + 1)):
        env.render()
        # generate random action
        action_random = a_random.eval({in_s: observation, in_s_prev: observation_prev})
        action_mu = action_random[0]
        action_sigma = action_random[1]
        running_noise = NOISE_DECAY * running_noise + (1 - NOISE_DECAY) * np.random.normal(size=out_dimen)
        action = action_mu #+ running_noise * action_sigma
        true_action = action
        if GAME == 'CartPole-v0':
            true_action = 0 if action < 0.5 else 1
        if step % 20 == 0:
            print(action_mu, action_sigma)
        # interact with environment
        temp_samples.append({in_s: observation, in_s_prev: observation_prev, in_action: action})
        observation_prev = observation
        observation, reward, done, info = env.step(true_action)
        reward = min(reward, 10)
        reward = max(reward, -10)
        rewards.append(reward)
        total_reward += reward
        if done:
            break
    for i in range(len(rewards)):
        mult = 1
        reward = 0
        sum_mults = 1e-8
        for j in range(len(rewards) - i):
            reward += rewards[i + j] * mult
            sum_mults += mult
            mult *= DECAY_FACTOR
        if not done:
            reward /= sum_mults
        temp_samples[i][in_reward] = reward
        train_samples.append(temp_samples[i])
    # train
    random.shuffle(train_samples)
    for ts in train_samples[:1000]:
        train_op.run(ts)
    train_samples = train_samples[:10000]
    print("Epoch : %3d  |  Reward : %5.0f  " % (epoch_i+1, total_reward) )


