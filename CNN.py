import CargaDatos as cd  # Import a module named "CargaDatos" as "cd"
import tensorflow as tf  # Import TensorFlow library as "tf"
import sklearn as sk  # Import scikit-learn library as "sk"
import keras  # Import Keras library
import matplotlib.pyplot as plt  # Import matplotlib.pyplot for plotting
from sklearn import model_selection

# Load data from the "CargaDatos" module into variables

X_images,y_images = cd.X_images,cd.y_images


X_train,X_test,y_train,y_test = model_selection.train_test_split(X_images,y_images,test_size=0.2,random_state=42)

X_train,X_val,y_train,y_val = model_selection.train_test_split(X_train,y_train,test_size=0.2,random_state=42)

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

# Define a class "NeuralModel" to create and train a neural network

class NeuralModel:
    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val, epochs, batch_size):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = keras.models.Sequential()  # Create a Sequential Keras model
        self.history = None
        self.accuracy = None
        self.loss = None
        self.val_accuracy = None
        self.val_loss = None

    def createModel(self):
        self.model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(332, 332,1)))
        self.model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Dropout(0.25))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def trainModel(self):
        self.history = self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                                       validation_data=(self.X_val, self.y_val),
                                         callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    
    def evaluateModel(self):
        self.accuracy = self.history.history['accuracy']
        self.loss = self.history.history['loss']
        self.val_accuracy = self.history.history['val_accuracy']
        self.val_loss = self.history.history['val_loss']
    
    def plotModel(self):
        plt.plot(self.accuracy)
        plt.plot(self.val_accuracy)
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

        plt.plot(self.loss)
        plt.plot(self.val_loss)
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

    def testModel(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        print('\nTest accuracy:', test_acc)
        print('\nTest loss:', test_loss)

# Usage:

model = NeuralModel(X_train, y_train, X_test, y_test, X_val, y_val, 10, 32)
model.createModel()
model.trainModel()
model.evaluateModel()
model.plotModel()
model.testModel()



