import numpy as np

def psnr(original_img, processed_img):
    # Convert images to numpy arrays
    original_arr = np.array(original_img).astype(np.float32)
    processed_arr = np.array(processed_img).astype(np.float32)

    # Calculate MSE
    mse = np.mean((original_arr - processed_arr) ** 2)

    # Calculate MAX (assuming pixel values are in the range [0, 255])
    max_pixel = 255

    # Calculate PSNR
    psnr_val = 10 * np.log10((max_pixel ** 2) / mse)

    return psnr_val

# Assuming test_low_light_images and test_high_light_images are lists of file paths
psnr_values = []

for low_image_file, high_image_file in zip(test_low_light_images, test_high_light_images):
    original_image = Image.open(high_image_file)  # Using high-light image as the reference
    enhanced_image = infer(Image.open(low_image_file))
    psnr_val = psnr(original_image, enhanced_image)
    psnr_values.append(psnr_val)

average_psnr = np.mean(psnr_values)
print("Average PSNR:", average_psnr)
