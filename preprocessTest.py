import glob
import os
import PIL
import numpy as np
import pickle
import sys
from PIL import Image
from preprocessing import preprocess 
import cv2
import argparse

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
    # img.show()
    return img

image_size = 100
print ("-------- PREPROCESS TESTING DATASET --------")

for dir in glob.glob("dataset_test/*"):
    directory = os.path.basename(dir)
    print ("--------" + directory + "--------")

    for file in glob.glob(dir + "/*.jpg"):
        
        # img = cv2.imread(file)
        # save_path = 'dataset_test/testing/' + directory + '/' + os.path.basename(file) + '.jpg'
        # img = preprocess(img, args.preprocessing, save_path, image_size)

        # testing_dataset.append(img)
        img = Image.open(file)
        img = crop(img)
        img = img.resize((100,100), Image.ANTIALIAS)
        img = np.array(img) 
        img = img[:, :, ::-1].copy() 

        # if directory == "Ceplok":
        save_path = 'result_preprocess_test/n/' + directory + '/' + os.path.basename(file) + '.jpg'
        preprocess(img, 'n', save_path, image_size)

        save_path = 'result_preprocess_test/g/' + directory + '/' + os.path.basename(file) + '.jpg'
        preprocess(img, 'g', save_path, image_size)

        save_path = 'result_preprocess_test/h/' + directory + '/' + os.path.basename(file) + '.jpg'
        preprocess(img, 'e', save_path, image_size)
        
        save_path = 'result_preprocess_test/c/' + directory + '/' + os.path.basename(file) + '.jpg'
        preprocess(img, 'c', save_path, image_size)
