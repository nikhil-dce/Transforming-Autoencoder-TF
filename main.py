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
tf.app.flags.DEFINE_integer('num_epochs', 0, "Number of epochs to train")
tf.app.flags.DEFINE_integer('num_gpus', 1, "Number of gpus to use")
tf.app.flags.DEFINE_integer('batch_size', 0, "Batch size")
tf.app.flags.DEFINE_integer('save_checkpoint_every', 0, "Save prediction after save_checkpoint_every epochs")
tf.app.flags.DEFINE_integer('save_pred_every', 0, "Save prediction after save_pred_every epochs"
)
tf.app.flags.DEFINE_integer('save_checkpoint_after', 0, "Save prediction after epochs")

FLAGS = tf.app.flags.FLAGS

def main():
    parser = argparse.ArgumentParser(description='Transforming Autoencoder')

    parser.add_argument('--num-epochs', type=int, default=800)
    parser.add_argument('--num-capsules', type=int, default=60)
    parser.add_argument('--generator-dimen', type=int, default=20)
    parser.add_argument('--recognizer-dimen', type=int, default=10)
    parser.add_argument('--save-pred-every', type=int, default=20)
    parser.add_argument('--save-checkpoint-every', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=100)
    
    #parser.add_argument('-chk', '--resume_checkpoint')
    args = parser.parse_args()
    print args

    FLAGS.num_epochs = args.num_epochs
    FLAGS.save_checkpoint_after = args.save_checkpoint_every
    FLAGS.save_pred_every = args.save_pred_every
    FLAGS.batch_size = args.batch_size
    
    train_images = load_train_data()
    X_trans, trans, X_original = translate(train_images)

    model = Model_Train(X_trans, trans, X_original, args.num_capsules, args.recognizer_dimen, args.generator_dimen, X_trans.shape[1])
    model.train()

if __name__ == "__main__":
    main()

    
    
