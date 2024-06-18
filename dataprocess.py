import os
import cv2
import numpy as np
from PIL import Image\
import tensorflow as tf

def load_data(image_path):
    print(f"Processing image: {image_path}")  # Debug statement
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, size=[IMAGE_SIZE, IMAGE_SIZE])
    image = image / 255.0
    return image

def data_generator(image_paths):
    # Ensure the paths are strings
    image_paths = [str(path) for path in image_paths]
    print("First few image paths:", image_paths[:5])  # Debug statement

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

def take(file_path):
    image = Image.open(file_path).convert('RGB')  # Ensure the image is in RGB format
    image = image.resize((256, 256))  # Resize to a fixed size if needed
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image

def preprocess(image_files):
    filtered_files = [file for file in image_files if not os.path.basename(file).startswith('.')]
    return [take(file) for file in filtered_files]