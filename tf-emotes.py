import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
import tensorflow as tf
import numpy as np
import sys
import glob
import time

class_names = [dir.split('\\')[1] for dir in glob.glob('./training_data/*/')]
print(class_names)

args = sys.argv[1:]
name = args[0]
files = os.listdir('./images/')
sorted = [file for file in files if name in file]

try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

model = keras.models.load_model('./models')

def process_img(img):
    i = keras.preprocessing.image.load_img(img, target_size=(128, 128))
    img_array = keras.preprocessing.image.img_to_array(i)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

for emote in sorted:
    current = process_img('./images/{}'.format(emote))
    predictions = model.predict(current)
    score = tf.nn.softmax(predictions[0])
    msg = "{} most likely belongs to {} with a {:.2f}% confidence.\n".format(
        emote, class_names[np.argmax(score)], 100 * np.max(score))
    print(msg)
    time.sleep(1)