import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
assert tf.__version__.startswith('2')
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import json
import time 
# define a class that load and preprocess one image
def load_process_image(file_path):

  # Load image (in PIL image format by default)
  image_original = load_img(file_path, target_size=(224, 224))
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

# 0 - Animal 1 - Nitrogen_Dioxide  2- Plant 3-circuit 4-earth
def get_result(predictions):
  return np.where(predictions == np.amax(predictions))

# load model
model = tf.keras.models.load_model('sim model\saved_model')

# Check its architecture
model.summary()
inputpath = 'input_images'
start_inference = False
stop = False

predictions = np.empty([1,5], dtype=int)
while(True):
    # List all files in a directory using os.listdir
    
    for entry in os.listdir(inputpath):
        if os.path.isfile(os.path.join(inputpath, entry)):
            if(entry == "done.jpg"):
                start_inference = True
            if(entry == "stop.txt"):
                stop = True

    if(stop):
        break
    
    if(start_inference):
        for entry in os.listdir(inputpath):
            if os.path.isfile(os.path.join(inputpath, entry)):
                image_file_path = "input_images\\{entry}".format(entry=entry)
                
                image_original, image_preprocessed = load_process_image(image_file_path)
                os.remove(image_file_path)
                predictions = model.predict(image_preprocessed)
                # result = get_result(predictions[0])
                print(predictions)
                mostprobable = 0
                for i in range(4):
                    if predictions[0][i] < predictions[0][i+1]:
                        mostprobable = i+1
                print(mostprobable)
                responseJson = {'class' : mostprobable}
                with open('input_images\\response.json', 'w') as json_file:
                    json.dump(responseJson, json_file)
        start_inference = False

