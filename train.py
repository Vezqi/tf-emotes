import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import PIL
import pathlib
import glob
import numpy
import string
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras
from PIL import Image
from pixelmatch import pixelmatch

d_dir = pathlib.Path('./training_data')

image_width = 256
image_height = 256

# https://stackoverflow.com/questions/45040145/unpredictable-cudnn-status-not-initialized-on-windows

try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    d_dir, validation_split=0.2, subset='training', seed=123, image_size=(image_width, image_height), batch_size=32)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    d_dir, validation_split=0.2, subset='validation', seed=123, image_size=(image_width, image_height), batch_size=32)

class_names = train_ds.class_names
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip(
            "horizontal", input_shape=(image_width, image_height, 3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(
        1./255, input_shape=(image_width, image_height, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'), # Change to 128 if  broken?
    layers.Dense(num_classes)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True), metrics=['accuracy'])

epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

model.summary()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('./models/')
input('Press any key to continue...')
