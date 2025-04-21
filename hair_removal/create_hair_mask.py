import os
import cv2

ORIGINAL_PATH = "./hair_removal/original_images/"
HAIR_PATH = "./hair_removal/hair_images/"
HAIR_MASK_PATH = "./hair_removal/hair_segmentation"

original_dir = os.listdir(ORIGINAL_PATH)
hair_dir = os.listdir(HAIR_PATH)
hair_mask_dir = os.listdir(HAIR_MASK_PATH)

for original_file, hair_file in zip(original_dir, hair_dir):
    original_image = cv2.imread(ORIGINAL_PATH + original_file, cv2.IMREAD_COLOR_RGB)
    hair_image = cv2.imread(HAIR_PATH + hair_file, cv2.IMREAD_COLOR_RGB)

    gray_original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    gray_hair_image = cv2.cvtColor(hair_image, cv2.COLOR_RGB2GRAY)

    diff = cv2.absdiff(gray_hair_image, gray_original_image)
    _, hair_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

    cv2.imwrite(f"mask_{original_file}", hair_mask)