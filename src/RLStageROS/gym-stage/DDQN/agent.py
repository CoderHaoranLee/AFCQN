import sys
import numpy as np
import keras.backend as K
import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Flatten, Reshape, LSTM, Lambda, Conv2D, GlobalMaxPool2D
from keras.regularizers import l2
from utils.networks import conv_block

class Agent:
    """ Agent Class (Network) for DDQN
    """

    def __init__(self, state_dim, action_dim, lr, tau, dueling):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        # Initialize Deep Q-Network
        self.model = self.network(dueling)
        self.model.compile(Adam(lr), loss={'q_value': 'mse'})
        # Build target Q-Network
        self.target_model = self.network(dueling)
        self.target_model.compile(Adam(lr), loss={'q_value': 'mse'})
        self.target_model.set_weights(self.model.get_weights())

    def huber_loss(self, y_true, y_pred):
        return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1, axis=-1)

    def network(self, dueling):
        """ Build Deep Q-Network
        """
        inp = Input((self.state_dim[1:]))
        x = conv_block(inp, 32, (2, 2), 8)
        x = conv_block(x, 64, (2, 2), 4)
        x = conv_block(x, 64, (2, 2), 3)
        q_x = Conv2D(1, kernel_size=1, padding='same', activation='linear')(x)
        global_features = conv_block(x, 64, (2, 2), 3)
        global_features = GlobalMaxPool2D()(global_features)
        # q_x = Flatten()(q_x)
        # global_features = Flatten()(global_features)
        global_x = Dense(self.action_dim+1, activation='linear')(global_features) # +1 for advantage value
        # q_x = keras.layers.concatenate([global_x, q_x], axis=-1)
        q_x = global_x

        if(True):
            # Have the network estimate the Advantage function as an intermediate layer
            # x = Dense(self.action_dim + 1, activation='linear')(x)
            q_x = Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True, axis=-1), output_shape=(self.action_dim,), name='q_value')(q_x)
        else:
            q_x = Dense(self.action_dim, activation='linear')(q_x)
        return Model(inp, q_x)

    def transfer_weights(self):
        """ Transfer Weights from Model to Target at rate Tau
        """
        W = self.model.get_weights()
        tgt_W = self.target_model.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.target_model.set_weights(tgt_W)

    def fit(self, inp, targ):
        """ Perform one epoch of training
        """
        return self.model.fit(self.reshape(inp), targ, epochs=1, verbose=0)

    def predict(self, inp):
        """ Q-Value Prediction
        """
        return self.model.predict(self.reshape(inp))

    def target_predict(self, inp):
        """ Q-Value Prediction (using target network)
        """
        return self.target_model.predict(self.reshape(inp))

    def reshape(self, x):
        if len(x.shape) < 4 and len(self.state_dim) > 2: return np.expand_dims(x, axis=0)
        elif len(x.shape) < 3: return np.expand_dims(x, axis=0)
        else: return x
