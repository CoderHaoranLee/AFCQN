import sys
import numpy as np
import keras.backend as K
import keras
from keras.models import Model
from keras.optimizers import Adam
import random
import tensorflow as tf
class Agent:
    """ Agent Class (Network) for DDQN
    """

    def __init__(self, state_dim, action_dim, tau, h_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.input = tf.placeholder(tf.float32, [None, self.state_dim[0], self.state_dim[1], self.state_dim[2]])

        x = tf.layers.conv2d(input, filters=32, kernel_size=8, padding='same', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, padding='same', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        x = tf.layers.conv2d(x, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        # x -> (batch_size*trainLength)x5 x 8 x 64

        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.trainLength = tf.placeholder(dtype=tf.int32)

        # === LSTM layer ===
        lstm_input = tf.reshape(tf.layers.flatten(x), [self.batch_size, self.trainLength, 5*8*64])
        lstm = tf.contrib.rnn.BasicLSTMCell(h_size, state_is_tuple=True)
        self.state_in = lstm.zero_state(self.batch_size, tf.float32)
        lstm_output, lstm_state_output = tf.nn.dynamic_rnn(
            inputs=lstm_input,
            cell=lstm,
            initial_state=self.state_in
        )
        lstm_output = tf.reshape(lstm_output, [-1, h_size])
        q_x = tf.layers.dense(lstm_output, self.action_dim + 1)
        q_x = tf.reshape(q_x, [self.batch_size, self.trainLength, self.action_dim+1])
        # === end ===

        self.trace_action_value = tf.expand_dims(q_x[:, :, 0], axis=-1) + q_x[:, :, 1:] - tf.reduce_mean(q_x[:, :, 1:], keepdims=True, axis=-1)
        self.network_output = tf.reshape(self.trace_action_value, [-1, self.action_dim])
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32) # (batch * trainLength) x action_dim
        self.loss = tf.reduce_sum(tf.square(self.targetQ - self.network_output))
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


class Experiments_buff:
    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size
        # the element of buffer is array with size (N x 5),
        # N is episode length, 5 is old_s, a, r, new_s, done.

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        tmp_buffer = [b for b in self.buffer if len(b)>=trace_length]
        sampled_episodes = random.sample(tmp_buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point:point + trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, 5])