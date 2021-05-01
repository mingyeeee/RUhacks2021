# Runs inference on one image. This is just a test program
# Requirements - TF 2.4.1
# Change image file path on line 44

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
assert tf.__version__.startswith('2')
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# define a class that load and preprocess one image
def load_process_image(file_path):

  # Load image (in PIL image format by default)
  image_original = load_img(file_path, target_size=(96, 96))
  print("Image size after loading", image_original.size)

  # Convert from numpy array
  image_array = img_to_array(image_original)
  print("Image size after converting to numpy array", image_array.shape)

  # Expand dims to add batch size as 1
  image_batch = np.expand_dims(image_array, axis=0)
  print("Image size after expanding dimension", image_batch.shape)

  # Preprocess image
  image_preprocessed = tf.keras.applications.vgg16.preprocess_input(image_batch)

  return image_original, image_preprocessed

# 0 - hydrogen | 1 - NO2 | 2 - oxygen
def get_result(predictions):
  return np.where(predictions == np.amax(predictions))

# load model
model = tf.keras.models.load_model('saved_model')

# Check its architecture
model.summary()

image_file_path = "C:\\Users\\mingy\\Documents\\RUhacks\\n02\\image6.jpg"
image_original, image_preprocessed = load_process_image(image_file_path)

predictions = model.predict(image_preprocessed)

result = get_result(predictions[0])
print(result[0])

