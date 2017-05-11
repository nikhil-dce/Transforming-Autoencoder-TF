import tensorflow as tf
import numpy as np
import os

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']="3"

from utils import load_data, translate
from train_model import Model_Train

TRAIN_DIR = '/media/data_raid/nikhil/encoder_summary/'
tf.app.flags.DEFINE_string('train_dir', TRAIN_DIR, """Directory where we write logs and checkpoints""")

NUMBER_OF_CAPSULES = 60
RECOGNISE_DIMEN = 10
GENERATE_DIMEN = 20
IN_DIMEN = 28*28

if __name__ == "__main__":

    train_images, train_labels = load_data()
    X_trans, trans, X_original = translate(train_images)

    model = Model_Train(X_trans, trans, X_original, NUMBER_OF_CAPSULES, RECOGNISE_DIMEN, GENERATE_DIMEN, IN_DIMEN)
    model.train()

    
    
