import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from skimage.io import imsave
import requests
from dataprocess import preprocess
import tensorflow as tf
from PIL import Image

def download_model_from_drive(drive_link, destination):
    """Download a file from Google Drive using the link."""
    file_id = drive_link.split('/')[-2]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    with open(destination, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded model from Google Drive: {destination}")

def load_images(image_dir):
    """Load images from the given directory."""
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    return image_paths

def save_images(images, filenames, output_dir):
    """Save images to the given directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for img, filename in zip(images, filenames):
        save_path = os.path.join(output_dir, filename)
        imsave(save_path, img)

def main():
    # Define paths
    test_input_dir = './test/low/'
    test_output_dir = './test/predicted/'
    model_path = 'model.h5'  # Path to save the downloaded model

    # Google Drive link for your model
    with open('weights_download_link.txt', 'r') as f:
        drive_link = f.read().strip()

    # Download the model
    download_model_from_drive(drive_link, model_path)

    # Load the trained model
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Load test images
    test_image_paths = load_images(test_input_dir)
    print(f"Found {len(test_image_paths)} test images.")

    # Process test images
    test_images = preprocess(test_image_paths)
    test_images = np.array(test_images)
    print("Test images processed.")

    # Predict denoised images
    predicted_images = model.predict(test_images)
    print("Prediction completed.")

    # De-normalize the images if necessary (converting back to original scale)
    predicted_images = (predicted_images * 255).astype('uint8')

    # Save the predicted images
    save_images(predicted_images, [os.path.basename(p) for p in test_image_paths], test_output_dir)
    print(f"Predicted images saved to {test_output_dir}")

if __name__ == "__main__":
    main()



