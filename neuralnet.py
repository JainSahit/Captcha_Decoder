import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

wd = "/Users/sahitjain/Desktop/Fall-19/IFT-598/Project/Data/"
path = wd + "/cleansed_characters/"


def load_images_from_folder(folder):
    images = []
    pic_target = []
    for imagename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, imagename), 0)
        if img is not None:
            pic_target.append(ord(imagename[:1]))
            images.append(img)
    return images, pic_target


X, y = load_images_from_folder(path)
X = np.array(X).reshape(-1, 40, 20, 1)
y = np.float32(y).reshape(3777, 1)
yDict = {i: y[i] for i in range(0, len(y))}
yDict = np.array(yDict)
X = X / 255.0
print(X.shape)
print(yDict)
model = Sequential()
# 3 convolutional layers
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

# 2 hidden layers
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(64))
model.add(Activation("relu"))

# The output layer with 13 neurons, for 13 classes
model.add(Dense(17))
model.add(Activation("softmax"))

# Compiling the model using some basic parameters
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

history = model.fit(X, y, batch_size=1, epochs=1, validation_split=0.0)

print(history.history.keys())
print(history.history)
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')