import numpy as np 
import cv2
import matplotlib.pyplot as plt
import itertools
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from PIL import Image

def crop(img):
    half_the_width = img.size[0]/2 
    half_the_height = img.size[1]/2
    img = img.crop(
        (
            half_the_width - half_the_height,
            half_the_height - half_the_height,
            half_the_width + half_the_height,
            half_the_height + half_the_height 
        )
    )
    return img

def preprocess(img, method, directory, image_size):
    if method == 'h':
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
        img = cv2.resize(img, (image_size, image_size)) 
        img = cv2.equalizeHist(img)
        img =np.asarray(img, dtype='float64')
        Image.fromarray(img).convert('RGB').save(directory)
        img = np.reshape(img, (image_size,image_size,1))
    elif method == 'g':
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
        img = cv2.resize(img, (image_size, image_size)) 
        img =np.asarray(img, dtype='float64')
        Image.fromarray(img).convert('RGB').save(directory)
        img = np.reshape(img, (image_size,image_size,1))
    elif method == 'c':
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
        img = cv2.resize(img, (image_size, image_size)) 
        # img = cv2.Canny(img,100,200)
        img = cv2.Canny(img,100,200)
        # img = cv2.Canny(img,255,255/3)
        Image.fromarray(img).convert('RGB').save(directory)
        img = np.asarray(img, dtype='float64')
        
        img = np.reshape(img, (image_size,image_size,1))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (image_size, image_size)) 
        Image.fromarray(img).convert('RGB').save(directory)
        img = np.asarray(img, dtype='float64')
        img = np.reshape(img, (image_size,image_size, 3))
    return img

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def CNN(network):
    network = conv_2d(network, 32, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 32, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 512, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 512, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 3, activation='softmax')

    network = regression(network, optimizer='adam',
                        loss='categorical_crossentropy',
                        learning_rate=0.001
                        )
    return network