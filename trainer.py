from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from utils import CNN 

import pickle
import tensorflow as tf
import argparse


parser = argparse.ArgumentParser(description='runn id')
parser.add_argument('id', type=str, help='run id')
args = parser.parse_args()


dim, X, Y, X_test, Y_test = pickle.load(open("data.pkl", "rb"))

X, Y = shuffle(X, Y)
image_size = 100


img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

network = input_data(shape=dim,
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug,
                     dtype=tf.float32)
network = CNN(network)        


model = tflearn.DNN(
    network, 
    tensorboard_verbose= 3, 
    checkpoint_path= 'checkpoint/batik-classifier.tfl.ckpt', 
    tensorboard_dir= "logs"
)
  
model.fit(X, Y, n_epoch=200, shuffle=True, 
        validation_set=(X_test, Y_test),
        show_metric=True, 
        snapshot_epoch=True,
        run_id=args.id)
                
model.save("checkpoint/batik-classifier.tfl")
