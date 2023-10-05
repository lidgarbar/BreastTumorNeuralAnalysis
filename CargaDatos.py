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

def loadX(tipo, dir_tipo):
    images = []

    # Obtener la lista de nombres de las imagenes de los tumores SIN LOS MASK
    image_files = [f for f in os.listdir(dir_tipo) if f.endswith(".png") and not f.endswith("_mask.png")]

    for image_filename in image_files:
        # Obtener el nombre base de la imagen (sin la extensión)
        base_name = os.path.splitext(image_filename)[0]
        
        # Cargar la imagen
        img = cv.imread(os.path.join(dir_tipo, image_filename), 0)

        # Inicializar una máscara vacía
        img_mask = np.zeros_like(img)

        # Buscar todas las máscaras correspondientes a esta imagen
        mask_files = [f for f in os.listdir(dir_tipo) if f.startswith(base_name + "_mask")]
       

        for mask_filename in mask_files:
            # Cargar la máscara
            mask = cv.imread(os.path.join(dir_tipo, mask_filename), 0)

            # Agregar la máscara a la imagen de la máscara
            img_mask = cv.bitwise_or(img_mask, mask)

        # Redimensionar la imagen y la máscara si es necesario
        img_resized = cv.resize(img, (332, 332), interpolation=cv.INTER_AREA)
        img_mask_resized = cv.resize(img_mask, (332, 332), interpolation=cv.INTER_AREA)

        # Aplicar la máscara a la imagen
        result = cv.bitwise_and(img_resized, img_mask_resized)

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
        #Sizes of each type of image
        x_benign =loadX('benign',self.dir_benign)
        x_malignant =loadX('malignant',self.dir_malignant)
        size_benign = len(x_benign)
        size_malignant = len(x_malignant)
        # Load data from the CSV file into a pandas DataFrame
        self.x_images = np.concatenate((x_benign,x_malignant),axis=0)
        

        # Create y vector
        self.y_images = np.concatenate((np.zeros(size_benign),np.ones(size_malignant)),axis=0)
        #haz un plot de una imagen con la etiqueta correspondiente
        plt.imshow(self.x_images[5],cmap="gray")
        plt.show()
        print(self.y_images[5])


ImageLoader = DataLoader_CNN()
ImageLoader.load_data()

