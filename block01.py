from sklearn.model_selection import train_test_split
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image_data = []
labels = []

for dirname, _, filenames in os.walk('D:\Documents\B.Tech\B.Tech_Third_Year\Sem02\Machine_Learning\Assignment05\images'):
    for filename in filenames:
        img = cv.imread(os.path.join(dirname, filename))
        image_data.append(img)
        
        # Parse the label from the filename and add it to the labels list
        label = filename.split('_')[0]
        if label[0]=='k':
            binary_label = 1
        else:
            binary_label = 0
        labels.append(binary_label)


image_data = np.array(image_data)
labels = np.array(labels)

# print(image_data.shape)
print(labels)

# Splitting into training and test data
train_data, test_data, train_labels, test_labels = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# split the training set further into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# print(type(train_labels))
# print(train_data.shape)



from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.metrics import *
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import time


num_epochs = 2
batch_size = 32

# Define the model architecture: VGG16(Block1)
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# compile the model
# sgd = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])




# # create a new TensorBoard callback
# log_dir = "./logs"
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# train the model
t1 = time.time()
model.fit(train_data, train_labels, epochs=num_epochs, validation_data=(val_data, val_labels))
t2 = time.time()
print('Time taken in training:',t2-t1,'seconds')

# evaluate the model on the test set
loss, acc = model.evaluate(test_data, test_labels)

print('Test Loss: {:.2f}, Test Accuracy: {:.2f}%'.format(loss, acc*100))
print(model.summary())