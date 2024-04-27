import cv2
import os
import argparse
import numpy as np
from scipy.ndimage import label
from PIL import Image


def pred_process(img):
    border_width = 4
    img[:border_width, :] = 0
    img[-border_width:, :] = 0
    img[:, :border_width] = 0
    img[:, -border_width:] = 0
    
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    
    _, threshold = cv2.threshold(blurred_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(img)
    cv2.drawContours(contour_img, contours, -1, (255), thickness=cv2.FILLED)
    
    labeled_array, num_features = label(contour_img)

    # Find the largest object
    # We skip the first element of the bincount because it represents the background (0 label)
    largest_object = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1

    # Create an array where only the largest object is white, and everything else is black
    largest_object_array = np.where(labeled_array == largest_object, 255, 0).astype(np.uint8)

    # Convert back to image
    largest_object_image = Image.fromarray(largest_object_array)

    return largest_object_image

def main(input_dir, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the input directory and save the result in the output directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('_output_ens.jpg'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # Read the image in grayscale
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Process the image
                processed_img = pred_process(img)
                # Save the processed image to the output directory
                processed_img.save(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images from an input directory and save them to an output directory.')
    parser.add_argument('--input', type=str, default="/root/1", help='Input directory path')
    parser.add_argument('--output', type=str, default="/root/2", help='Output directory path')

    args = parser.parse_args()
    main(args.input, args.output)
