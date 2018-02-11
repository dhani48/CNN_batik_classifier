  # -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import glob
import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from sklearn.metrics import confusion_matrix
from preprocessing import plot_confusion_matrix
from tflearn.layers.normalization import local_response_normalization

import tensorflow as tf
import scipy
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from utils import preprocess, CNN, crop



parser = argparse.ArgumentParser(description='decide if bird or not')
parser.add_argument('id', type=str, help='preprocessing code')
args = parser.parse_args()



img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

network = input_data(shape=[None, 100, 100, 1],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

network = CNN(network)

model = tflearn.DNN(network)

model.load("checkpoint/batik-classifier.tfl")

# img = scipy.ndimage.imread(args.image, mode="RGB")
# img = Image.open(args.image)
#  img = scipy.misc.imresize(img, (100, 100), interp="bicubic").astype(np.float32, casting='unsafe')
# img = crop(img)
# img = img.resize((100,100), Image.ANTIALIAS)

# prediction = model.predict([img])
# print(prediction)
# tf.confusion_matrix
# print(np.argmax(prediction[0]))


y_test = []
y_pred = []


for dir in glob.glob("dataset_test/*"):         
    directory = os.path.basename(dir)               
    print(dir)               
    i = 0
    for file in glob.glob(dir + "/*.jpg"):
        img = Image.open(file)
        img = crop(img)
        img = img.resize((100,100), Image.ANTIALIAS)
        save_path = 'result_preprocess_test/' + args.id + '/' + directory + '/' + os.path.basename(file) + '.jpg'
        img = np.array(img) 
        img = img[:, :, ::-1].copy() 
        img = preprocess(img, args.id, save_path, 100)
        prediction = model.predict([img])
        print('BLAAA', prediction)
        y_pred.append(np.argmax(prediction))
        print(directory, prediction, "prediction: ", np.argmax(prediction[0]) )

        if directory == "Ceplok":
            y_test.append(0)
        elif directory == "Kawung":
            y_test.append(1)

        elif directory == "Parang":
            y_test.append(2)



matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(matrix, ['Ceplok', 'Kawung', 'Parang'], normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues)
plt.show()

# if is_bird:
#     print("a bird")
# else:
#     print("not a bird!")
