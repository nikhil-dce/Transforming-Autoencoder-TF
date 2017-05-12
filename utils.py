from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random

def load_train_data():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return mnist.train.images

def load_validation_data():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return mnist.validation.images

def translate(X_input, trans_range=5, shape=28):
    X = np.reshape(X_input, (len(X_input), shape, shape))
    X_trans, trans, X_original = [], [], []

    for i in np.random.permutation(len(X)):
        trans_x = random.randint(-trans_range, trans_range)
        trans_y = random.randint(-trans_range, trans_range)

        trans_img = np.roll(np.roll(X[i], trans_x, axis=0), trans_y, axis=1)
        X_trans.append(trans_img.flatten())
        X_original.append(X_input[i])
        trans.append((trans_x, trans_y))

    return np.array(X_trans), np.array(trans), np.array(X_original)

