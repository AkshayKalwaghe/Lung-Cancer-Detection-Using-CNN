# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 10:34:21 2021

@author: Dell
"""
### importing the libraries ####

import tensorflow as tf
import os
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


############  Train Dataset   ############
train_dir = os.path.join(r"C:\Users\akshaybk\Desktop\lung cancer\dataset_lung (1)\dataset\training") #Train directory

############  Test Dataset   ############
test_dir = os.path.join(r"C:\Users\akshaybk\Desktop\lung cancer\dataset_lung (1)\dataset\testing") # Test Directory


# Data Augmentation
# Global Variables
IMG_WIDTH = 224 #Width of image
IMG_HEIGHT = 224 # Height of Image
BATCH_SIZE = 32 # Batch Size 
train_data_size = 228 #
test_data_size = 54

train_data = ImageDataGenerator(
                rescale = 1./255, #normalizing the input image
                rotation_range = 30, # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0.2, # zooom the images
                brightness_range = (0.5, 1.5))

test_data = ImageDataGenerator(
                rescale = 1./255)

train_set = train_data.flow_from_directory(
                train_dir,
                target_size=(IMG_WIDTH,IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                class_mode = 'categorical')

test_set = test_data.flow_from_directory(
                test_dir,
                target_size = (IMG_WIDTH,IMG_HEIGHT),
                batch_size = BATCH_SIZE,
                shuffle=False,
                class_mode = 'categorical')

labels_values,no_of_images = np.unique(train_set.classes,return_counts = True)
dict(zip(train_set.class_indices,no_of_images))
labels = test_set.class_indices
labels = { v:k for k,v in labels.items() } # Flipping keys and values
values_lbl = list(labels.values()) # Taking out only values from dictionary

# Defining all layers.
dense = tf.keras.layers.Dense
conv = tf.keras.layers.Conv2D
max_pooling = tf.keras.layers.MaxPooling2D
flatten = tf.keras.layers.Flatten()
dropout = tf.keras.layers.Dropout(0.2)

# Sequential Model
model = tf.keras.Sequential()

# 1st layer
model.add(conv(16,(3,3),input_shape = (IMG_WIDTH,IMG_HEIGHT,3),padding='same',activation='relu'))
model.add(max_pooling(2,2))

# 2nd Layer
model.add(conv(16,(3,3),padding='same',activation='relu'))
model.add(max_pooling(2,2))

# 3rd Layer
model.add(conv(32,(3,3),padding='same',activation='relu'))
model.add(max_pooling(2,2))

# 4th Layer
model.add(conv(32,(3,3),padding='same',activation='relu'))
model.add(max_pooling(2,2))

# 5th Layer
model.add(conv(32,(3,3),padding='same',activation='relu'))
model.add(max_pooling(2,2))

# Flatten Layer
model.add(flatten)

# 1st Hidden Layer
model.add(dense(512,activation='relu',))
model.add(dropout)

# 2nd Hidden Layer
model.add(dense(256,activation='relu'))

# Output Layer
model.add(dense(2,activation='softmax'))

# Summary
model.summary()

# Compiling model
model.compile(loss='categorical_crossentropy',optimizer = tf.keras.optimizers.RMSprop(lr = 0.001),metrics=['acc'] )

history_model = model.fit_generator(train_set,epochs=100,
validation_data=test_set,verbose=1,
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience = 105)])

#Save our model
model.save("cancer_weights.h5")

# Accuracy comparison
plt.plot(history_model.history['acc'])
plt.plot(history_model.history['val_acc'])
plt.legend(['acc','val_acc'])
plt.show()
#
# Loss Comparison
plt.plot(history_model.history['loss'])
plt.plot(history_model.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.show()
