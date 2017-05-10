__author__ = "Nikhil Mehta"
__copyright__ = "--"

import numpy as np
import tensorflow as tf

class Capsule(object):

    def __init__(self, in_dim, r_dim, g_dim):

        self.in_dim = in_dim
        self.r_dim = r_dim
        self.g_dim = g_dim
    
    def get_fc_var(self, in_size, out_size, name):

        # TODO
        # Store this variable in CPU instead of GPU when multiple GPUs
        # with tf.device('/cpu:0')

        initial_value = tf.truncated_normal([in_size, out_size], .0, .001)
        weights = tf.get_variable(name=name + "_weights", initializer=initial_value)

        bias_initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = tf.get_variable(name=name + "_biases", initializer=bias_initial_value)

        return weights, biases

    def fc_layer(self, bottom, in_size, out_size, name):

        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            fc = tf.nn.bias_add(tf.matmul(bottom, weights), biases)

        return fc

    def build(self, X_in, extra_in):

        rec = tf.sigmoid(self.fc_layer(X_in, self.in_dim, self.r_dim, 'recog_layer_pre_act'), 'recog_layer')
        
        xy_vec = self.fc_layer(rec, self.r_dim, 2, 'xy_prediction')
        pro = tf.sigmoid(self.fc_layer(rec, self.r_dim, 1, 'probability_lin'), 'probability_prediction')
        probability_vec = tf.tile(pro, (1, self.in_dim))

        xy_extend = tf.add(xy_vec, extra_in)
        gen = tf.sigmoid(self.fc_layer(xy_extend, 2, self.g_dim, 'gen_pre_act'), 'gen_layer')

        out = self.fc_layer(gen, self.g_dim, self.in_dim, 'out_prediction')

        return tf.multiply(out, probability_vec)   
