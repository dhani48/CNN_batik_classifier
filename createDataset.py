import glob
import os
import PIL
import numpy as np
import pickle
import sys
from PIL import Image
from utils import preprocess 
import cv2
import argparse

#0 = Ceplok
#1 = Kawung
#2 = Parang
parser = argparse.ArgumentParser(description='choose preprocessing method')
parser.add_argument('preprocessing', type=str, help='what preprocessing to do? 0: none, 1: histogram eq, 2: grayscale, 3: edge')
args = parser.parse_args()



training_dataset = []
training_class = []
testing_dataset = []
testing_class = []

train_batch = open("data.pkl",'wb')
image_size = 100
print ("-------- CREATING TRAINING DATASET --------")

# traverse training dataset directory
for dir in glob.glob("dataset_batik/training/*"):         
     #r eturns directory name          
    directory = os.path.basename(dir)                              
    print ("--------" + directory + "--------")
    i = 0
    # traverse directory retrieved from the first loop
    for file in glob.glob(dir + "/*.jpg"):   

        # open images in the directory       
        # img= Image.open(file)    
        # resize image to 100x100 pixels 
        img = cv2.imread(file)
        print(args.preprocessing)
        save_path = 'result_preprocess/training/' + directory + '/batik' + str(i) + '.jpg'
        i+=1
        img = preprocess(img, args.preprocessing, save_path, image_size)

        # append image to a predefined array to contain the images
        
        training_dataset.append(img)

        # the labeling of the files based on the directory the image is in
        if directory == "Ceplok":
            print ("appending: " + file)
            print ("class: Ceplok")
            training_class.append([1,0,0])

        elif directory == "Kawung":
            print ("appending: " + file)
            print ("class: Kawung")
            training_class.append([0,1,0])

        elif directory == "Parang":
            print ("appending: " + file)
            print ("class: Parang")
            training_class.append([0,0,1])
        else:
            print ("errrr")
        print(training_class)

print ("-------- CREATING VALIDATION DATASET --------")

for dir in glob.glob("dataset_batik/testing/*"):
    directory = os.path.basename(dir)
    print ("--------" + directory + "--------")

    for file in glob.glob(dir + "/*.jpg"):
        
        img = cv2.imread(file)
        save_path = 'result_preprocess/testing/' + directory + '/' + os.path.basename(file) + '.jpg'
        
        img = preprocess(img, args.preprocessing, save_path, image_size)

        testing_dataset.append(img)

        if directory == "Ceplok":
            print ("appending: " + file)
            print ("class: Ceplok")
            testing_class.append([1,0,0])

        elif directory == "Kawung":
            print ("appending: " + file)
            print ("class: Kawung")
            testing_class.append([0,1,0])

        elif directory == "Parang":
            print ("appending: " + file)
            print ("class: Parang")
            
            testing_class.append([0,0,1])
       
        else:
            print ("errrr")

dim = []
if args.preprocessing == 'n':
    dim = [image_size, image_size, 3]
else:
    dim = [image_size, image_size, 1]


pickle.dump((dim, training_dataset, training_class, testing_dataset, testing_class),train_batch)
train_batch.close()
