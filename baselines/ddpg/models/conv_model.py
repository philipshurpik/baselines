import tensorflow as tf
import tensorflow.contrib as tc
import baselines.common.tf_util as U
from .model import Model


def make_conv_hidden_layers(x, action, layer_norm):
    x = tf.layers.conv2d(x, 32, kernel_size=[1, 8], strides=[1, 4], name='conv1')
    if layer_norm:
        x = tc.layers.layer_norm(x, center=True, scale=True)
    x = tf.nn.relu(x)

    x = tf.layers.conv2d(x, 64, kernel_size=[1, 4], strides=[1, 2], name='conv2')
    if layer_norm:
        x = tc.layers.layer_norm(x, center=True, scale=True)
    x = tf.nn.relu(x)

    x = U.flattenallbut0(x)
    if action is not None:
        x = tf.concat([x, action], axis=-1)

    x = tf.layers.dense(x, 256, kernel_initializer=U.normc_initializer(1.0), name='dense')
    if layer_norm:
        x = tc.layers.layer_norm(x, center=True, scale=True)
    x = tf.nn.relu(x)
    return x


class ActorConv(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(ActorConv, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = make_conv_hidden_layers(obs, action=None, layer_norm=self.layer_norm)
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class CriticConv(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(CriticConv, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = make_conv_hidden_layers(obs, action=action, layer_norm=self.layer_norm)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x
