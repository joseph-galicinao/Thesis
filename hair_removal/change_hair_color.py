import os
import cv2
import numpy as np

ORIGINAL_PATH = "./hair_removal/original_images/"
HAIR_MASK_PATH = "./hair_removal/hair_segmentation/"

original_dir = os.listdir(ORIGINAL_PATH)
hair_mask_dir = os.listdir(HAIR_MASK_PATH)

for original_file, hair_mask_file in zip(original_dir, hair_mask_dir):
    original_image = cv2.imread(ORIGINAL_PATH + original_file)

    hair_mask_image = cv2.imread(HAIR_MASK_PATH + hair_mask_file, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(hair_mask_image, 127, 255, cv2.THRESH_BINARY)

    mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Define the dark brown color in BGR
    dark_brown = np.array([130, 180, 230], dtype=np.uint8)
    brown_layer = np.full_like(original_image, dark_brown)

    recolored_hair = np.where(mask_3ch == 255, brown_layer, original_image)

    cv2.imwrite(f"blonde_{original_file}", recolored_hair)

