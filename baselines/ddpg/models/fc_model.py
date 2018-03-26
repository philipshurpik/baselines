import tensorflow as tf
import tensorflow.contrib as tc
from .model import Model

FC_LAYERS = 128

def make_fc_hidden_layers(x, action, layer_norm):
    x = tf.layers.dense(x, FC_LAYERS if action is None else FC_LAYERS // 2)
    if layer_norm:
        x = tc.layers.layer_norm(x, center=True, scale=True)
    x = tf.nn.relu(x)

    if action is not None:
        x = tf.concat([x, action], axis=-1)

    x = tf.layers.dense(x, FC_LAYERS if action is None else FC_LAYERS // 2)
    if layer_norm:
        x = tc.layers.layer_norm(x, center=True, scale=True)
    x = tf.nn.relu(x)
    return x


class ActorFC(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(ActorFC, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = make_fc_hidden_layers(obs, action=None, layer_norm=self.layer_norm)
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class CriticFC(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(CriticFC, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = make_fc_hidden_layers(obs, action=action, layer_norm=self.layer_norm)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x
