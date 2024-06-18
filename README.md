# Low Light Image Enhancement

This repository contains a project for enhancing low-light images using the PRIDNet model. The model has achieved a Peak Signal-to-Noise Ratio (PSNR) of 21.3, demonstrating its effectiveness in improving the quality of low-light images.

## Model
PRIDNet is a state-of-the-art deep learning model designed for low-light image enhancement. The model architecture effectively denoises and enhances images by progressively refining them through multiple stages.

## Dataset
The dataset for this project consists of low-light images for training and testing. The images are preprocessed and normalized before being fed into the model.

## Training
The model was trained using a set of low-light images. The training script, `train.py`, handles the data loading, preprocessing, model training, and saving the trained model weights.

## Testing
The testing script, `main.py`, reads the test images from the `./test/low/` directory, enhances them using the trained PRIDNet model, and saves the enhanced images in the `./test/predicted/` directory.

## Results
The PRIDNet model achieved a PSNR of 21.3 on the test set, indicating a significant improvement in image quality.




