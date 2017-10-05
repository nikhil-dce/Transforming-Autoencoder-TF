__author__ = "Nikhil Mehta"
__copyright__ = "--"

import tensorflow as tf
import numpy as np
import os
import sys
import argparse

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']="3"

from utils import load_train_data, load_validation_data, translate
from train_model import Model_Train

RUN = 'run_2'
TRAIN_DIR = '/data1/nikhil/trans-autoencoder-summary/'

tf.app.flags.DEFINE_string('train_dir', TRAIN_DIR+RUN, """Directory where we write logs and checkpoints""")
tf.app.flags.DEFINE_string('checkpoint_dir', TRAIN_DIR+RUN, """Directory from where to read the checkpoint""")
tf.app.flags.DEFINE_integer('num_epochs', 800, "Number of epochs to train")
tf.app.flags.DEFINE_integer('num_gpus', 1, "Number of gpus to use")
tf.app.flags.DEFINE_integer('batch_size', 100, "Batch size")
tf.app.flags.DEFINE_integer('save_checkpoint_every', 200, "Save prediction after save_checkpoint_every epochs")
tf.app.flags.DEFINE_integer('save_pred_every', 20, "Save prediction after save_pred_every epochs"
)
tf.app.flags.DEFINE_integer('save_checkpoint_after', 0, "Save prediction after epochs")
tf.app.flags.DEFINE_integer('num_capsules', 60, "Number of capsules")
tf.app.flags.DEFINE_integer('generator_dimen', 20, "Dimension of generator layer")
tf.app.flags.DEFINE_integer('recognizer_dimen', 10, "Dimension of recognition layer")

FLAGS = tf.app.flags.FLAGS

def main():

    train_images = load_train_data()
    X_trans, trans, X_original = translate(train_images)

    model = Model_Train(X_trans, trans, X_original, FLAGS.num_capsules, FLAGS.recognizer_dimen, FLAGS.generator_dimen, X_trans.shape[1])
    model.train()

if __name__ == "__main__":
    main()

    
    
