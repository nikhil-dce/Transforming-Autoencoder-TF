import tensorflow as tf
import numpy as np
import sys
import math

from capsule import Capsule

class TransformingAutoencoder:

    def __init__(self, in_dimen, r_dimen, g_dimen, num_capsules, batch_size, transforming_params=2):

        self.num_capsules = num_capsules
        self.in_dimen = in_dimen
        self.shape = int(math.sqrt(in_dimen))
        self.r_dimen = r_dimen
        self.g_dimen = g_dimen
        self.transforing_params = transforming_params

        # Need this param for tensorboard visualization
        self.batch_size = batch_size
        
    def forward_pass(self, X_in, extra_in):
        
        X_reshaped = tf.reshape(X_in, (self.batch_size, self.shape, self.shape))
        tf.summary.image('original', tf.expand_dims(X_reshaped[:,:,:], -1))
        
        capsules_out = []
        for i in range(self.num_capsules):
            with tf.variable_scope('capsule_%d' % (i)):
                capsule = Capsule(self.in_dimen, self.r_dimen, self.g_dimen)
                capsule_out = capsule.build(X_in, extra_in)
                capsules_out.append(capsule_out)

        all_caps_out = tf.add_n(capsules_out)
        X_prediction = tf.sigmoid(all_caps_out)

        X_prediction_reshaped = tf.reshape(X_prediction, (self.batch_size, self.shape, self.shape))
        tf.summary.image('prediction', tf.expand_dims(X_prediction_reshaped[:,:,:], -1))
        
        return X_prediction
        

    def loss(self, X_pred, X_out):

        batch_squared_error = tf.reduce_sum(tf.square(tf.subtract(X_pred, X_out)), axis=1)
        mse = tf.reduce_mean(batch_squared_error)
        return mse
