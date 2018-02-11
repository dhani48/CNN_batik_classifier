from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.normalization import local_response_normalization
import pickle
import tensorflow as tf

X, Y, X_test, Y_test = pickle.load(open("data.pkl", "rb"),encoding='latin1')

parser = argparse.ArgumentParser(description='runn id')
parser.add_argument('id', type=str, help='run id')
args = parser.parse_args()

X, Y = shuffle(X, Y)
image_size = 100

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

network = input_data(shape=[None, image_size, image_size, 1],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug,
                     dtype=tf.float32)

# model 1
# network = conv_2d(network, 32, 3, activation='relu')
# network = max_pool_2d(network, 2)
# network = conv_2d(network, 64, 3, activation='relu')
# network = conv_2d(network, 64, 3, activation='relu')
# network = max_pool_2d(network, 2)
# network = fully_connected(network, 512, activation='relu')
# network = dropout(network, 0.5)
# network = fully_connected(network, 3, activation='softmax')
# end model1


# model 'AlexNet'
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 3, activation='softmax')

# end model 'AlexNet'

# print(network)

network = regression(network, optimizer='adam',
                    loss='categorical_crossentropy',
                    learning_rate=0.001)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True, gpu_options=gpu_options)
# sess = tf.Session(config = config)


# init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# init = tf.global_variables_initializer()
# sess.run(init)

model = tflearn.DNN(
    network, 
    tensorboard_verbose= 3, 
    checkpoint_path= 'checkpoint/batik-classifier.tfl.ckpt', 
    tensorboard_dir= "logs"
)

tflearn.init_graph(gpu_memory_fraction=0.333,soft_placement=True)

                    
with tf.device('/gpu:0'):

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    # config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True, gpu_options=gpu_options)
    # sess = tf.Session(config = config)

    
    # init = tf.initialize_all_variables()
    # sess.run(init)
    # tflearn.is_training(True, session=sess)    
    with tf.contrib.framework.arg_scope(
        [tflearn.variables.variable], 
        device='/cpu:0'):
        model.fit(X, Y, n_epoch=30, shuffle=True, 
            validation_set=(X_test, Y_test),
            show_metric=True, 
            snapshot_epoch=True,
            run_id=args.id)
                
model.save("checkpoint/batik-classifier.tfl")
