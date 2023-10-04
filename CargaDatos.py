import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import model_selection
import os
import cv2 as cv
import matplotlib.pyplot as plt

class DataLoader_NN:
    def __init__(self, csv_path='./data/biopsy.csv'):
        self.csv_path = csv_path
        self.data = None
        self.x_biopsy = None
        self.y_biopsy = None
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None

    def load_data(self):
        # Load data from the CSV file into a pandas DataFrame
        self.data = pd.read_csv(self.csv_path, header=0)

    def preprocess_data(self):
        # Extract target values (diagnosis) and training features
        self.y_biopsy = self.data['diagnosis']
        self.x_biopsy = self.data.drop(['diagnosis'], axis=1)
        self.x_biopsy = np.array(self.x_biopsy)

        # Convert diagnosis values ('M' and 'B') to binary values (1 and 0)
        self.y_biopsy = self.y_biopsy.map({'M': 1, 'B': 0})
        self.y_biopsy = np.array(self.y_biopsy)

        # Normalize the training features using a class method
        self.x_biopsy = self.normalize_data(self.x_biopsy)

    @staticmethod
    def normalize_data(x):
        # Normalize data using mean and standard deviation
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        return (x - mean) / std

    def split_data(self):
        # Split the data into training, test, and validation sets
        X_train, self.X_test, y_train, self.y_test = sk.model_selection.train_test_split(self.x_biopsy, self.y_biopsy, test_size=0.2, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = sk.model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    def print_shapes(self):
        # Print the shapes of the training, test, and validation sets
        print("X_train: ", self.X_train.shape)
        print("X_test: ", self.X_test.shape)
        print("X_val: ", self.X_val.shape)

# Usage:
'''
loader = DataLoader_NN()
loader.load_data()
loader.preprocess_data()
loader.split_data()
loader.print_shapes()
'''
#TODO: HACER LO DEL MASK_1 y MASK CON LAS IMAGENES
def loadX(tipo,dir_tipo):
    images = []
    size = int(len(os.listdir(dir_tipo))/2)
    print(size)
    for i in range(size-1):
        dir= tipo+' ('+str(i+1)+').png'
        dir_mask= tipo+' ('+str(i+1)+')_mask.png'
        print(os.path.join(dir_tipo,dir))
        img = cv.imread(os.path.join(dir_tipo,dir),0)
        img_mask = cv.imread(os.path.join(dir_tipo,dir_mask),0)
        img_resized= cv.resize(img,(332, 332),interpolation=cv.INTER_AREA)
        img_mask_resized= cv.resize(img_mask,(332, 332),interpolation=cv.INTER_AREA)
        result = cv.bitwise_and(img_resized,img_mask_resized)
        images.append(result)
    X = np.array(images)
    return X

class DataLoader_CNN:
    def __init__(self, images_path='./data'):
        self.dir_benign=os.path.join(images_path,'benign')
        self.dir_malignant=os.path.join(images_path,'malignant')
        self.dir_normal=os.path.join(images_path,'normal')
        self.x_images = None
        self.y_images = None
        self.X_images_train = None
        self.X_images_test = None
        self.X_images_val = None
        self.y_images_train = None
        self.y_images_test = None
        self.y_images_val = None

    

    def load_data(self):
        # Load data from the CSV file into a pandas DataFrame
        size_benign= len(os.listdir(self.dir_benign))/2
        size_malignant= len(os.listdir(self.dir_malignant))/2
        size_normal= len(os.listdir(self.dir_normal))/2

        self.x_images = loadX('benign',self.dir_benign)+loadX('malignant',self.dir_malignant)
        +loadX('normal',self.dir_normal)
        #haz un plot de una imagen
        plt.imshow(self.x_images[0],cmap="gray")
        self.y_images = np.concatenate((np.zeros(size_benign),np.ones(size_malignant),np.full(size_normal,2)))

    




ImageLoader = DataLoader_CNN()
ImageLoader.load_data()

'''
size_benign= len(os.listdir(dir_benign))/2
    size_malignant= len(os.listdir(dir_malignant))/2
        size_normal= len(os.listdir(dir_normal))/2
        '''