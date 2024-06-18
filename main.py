import os
import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split



import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose,\
                                    GlobalAveragePooling2D, AveragePooling2D, MaxPool2D, UpSampling2D,\
                                    BatchNormalization, Activation, Flatten, Dense, Input,\
                                    Add, Multiply, Concatenate, concatenate, Softmax
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax

tf.keras.backend.set_image_data_format('channels_last')



from structure import create_model
from dataprocess import preprocess


model=create_model()


import tensorflow as tf
import gdown
import numpy as np
import cv2
import os

def download_weights_from_drive(drive_link, output_path):
    # Extract file ID from Google Drive link
    file_id = drive_link.split('/')[-2]
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    gdown.download(url, output_path, quiet=False)

# Load the pre-defined model (assume the model is already defined as 'model')
# Assume create_model() is defined elsewhere

# Google Drive link to the weights file
drive_link = 'https://drive.google.com/file/d/15Shh4AF1g2DD_c0I3vBn2QjRGhq8WNLh/view?usp=sharing'

# Define the output path for the weights file
output_directory = './weights'
os.makedirs(output_directory, exist_ok=True)
weights_path = os.path.join(output_directory, 'model_weights.h5')

# Download the weights file
download_weights_from_drive(drive_link, weights_path)

# Load the weights into the model
model.load_weights(weights_path)



