import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

horizontal_kernel = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8)
vertical_kernel = np.array([[0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [0]], dtype=np.uint8)
down_diagonal_kernel = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
up_diagonal_kernel = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

ROW = 3
COL = 4

HAIR_PATH = "./hair_removal/hair_images/blonde_hair/"
HAIR_MASK_PATH = "./hair_removal/hair_segmentation/"

hair_dir = os.listdir(HAIR_PATH)
hair_mask_dir = os.listdir(HAIR_MASK_PATH)

def dilate_image(image):
    horizontal_dilate = cv2.dilate(image, horizontal_kernel, iterations=1)
    vertical_dilate = cv2.dilate(image, vertical_kernel, iterations=1)
    up_diagonal_dilate = cv2.dilate(image, up_diagonal_kernel, iterations=1)
    down_diagonal_dilate = cv2.dilate(image, down_diagonal_kernel, iterations=1)

    dilated_image = cv2.bitwise_or(cv2.bitwise_or(horizontal_dilate, vertical_dilate),
                                   cv2.bitwise_or(down_diagonal_dilate, up_diagonal_dilate))
    
    return dilated_image

def erode_image(image):
    horizontal_erode = cv2.erode(image, horizontal_kernel, iterations=1)
    vertical_erode = cv2.erode(image, vertical_kernel, iterations=1)
    up_diagonal_erode = cv2.erode(image, up_diagonal_kernel, iterations=1)
    down_diagonal_erode = cv2.erode(image, down_diagonal_kernel, iterations=1)

    erode_image = cv2.bitwise_or(cv2.bitwise_or(horizontal_erode, vertical_erode),
                                   cv2.bitwise_or(down_diagonal_erode, up_diagonal_erode))
    
    return erode_image

def blackhat(image):
    blackhat_horizontal = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, horizontal_kernel, iterations=1)
    blackhat_vertical = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, vertical_kernel, iterations=1)
    blackhat_up_diag = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, up_diagonal_kernel, iterations=1)
    blackhat_down_diag = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, down_diagonal_kernel, iterations=1)

    blackhat_image = cv2.bitwise_or(cv2.bitwise_or(blackhat_horizontal, blackhat_vertical),
                                    cv2.bitwise_or(blackhat_up_diag, blackhat_down_diag))
    
    return blackhat_image

def tophat(image):
    tophat_horizontal = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, horizontal_kernel, iterations=2)
    tophat_vertical = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, vertical_kernel, iterations=2)
    tophat_up_diag = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, up_diagonal_kernel, iterations=2)
    tophat_down_diag = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, down_diagonal_kernel, iterations=2)

    tophat_image = cv2.bitwise_or(cv2.bitwise_or(tophat_horizontal, tophat_vertical),
                                    cv2.bitwise_or(tophat_up_diag, tophat_down_diag))
    
    return tophat_image

def close(image):
    close_horizontal = cv2.morphologyEx(image, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)
    close_vertical = cv2.morphologyEx(image, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    close_up_diag = cv2.morphologyEx(image, cv2.MORPH_CLOSE, up_diagonal_kernel, iterations=1)
    close_down_diag = cv2.morphologyEx(image, cv2.MORPH_CLOSE, down_diagonal_kernel, iterations=1)

    close_image = cv2.bitwise_or(cv2.bitwise_or(close_horizontal, close_vertical),
                                    cv2.bitwise_or(close_up_diag, close_down_diag))
    
    return close_image

def open(image):
    open_horizontal = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    open_vertical = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    open_up_diag = cv2.morphologyEx(image, cv2.MORPH_OPEN, up_diagonal_kernel, iterations=1)
    open_down_diag = cv2.morphologyEx(image, cv2.MORPH_OPEN, down_diagonal_kernel, iterations=1)

    open_image = cv2.bitwise_or(cv2.bitwise_or(open_horizontal, open_vertical),
                                    cv2.bitwise_or(open_up_diag, open_down_diag))
    
    return open_image

# Read the image
IOU = 0
for hair_file, hair_mask_file in zip(hair_dir, hair_mask_dir):
    hair_image = cv2.imread(HAIR_PATH + hair_file, cv2.IMREAD_COLOR_RGB)
    hair_mask_image = cv2.imread(HAIR_MASK_PATH + hair_mask_file, cv2.IMREAD_GRAYSCALE)

    #Gray scale
    grayscale_image = cv2.cvtColor(hair_image, cv2.COLOR_RGB2GRAY)

    # Edge Detection
    gaussian_image = cv2.GaussianBlur(grayscale_image, (7, 7), 0)
    canny_image = cv2.Canny(gaussian_image, threshold1=50, threshold2=150)

    # Dilate image
    # dilated_image = dilate_image(canny_image)

    # Black har image
    # blackhat_image = blackhat(canny_image)

    # Close image
    closed_image = close(canny_image)

    # Top hat image
    # tophat_image = tophat(closed_image)

    

    # Black hat filter to detect dark hairs
    # tophat_horizontal = cv2.morphologyEx(canny_image, cv2.MORPH_TOPHAT, horizontal_kernel, iterations=1)
    # tophat_vertical = cv2.morphologyEx(canny_image, cv2.MORPH_TOPHAT, vertical_kernel, iterations=1)
    # tophat_up_diag = cv2.morphologyEx(canny_image, cv2.MORPH_TOPHAT, up_diagonal_kernel, iterations=1)
    # tophat_down_diag = cv2.morphologyEx(canny_image, cv2.MORPH_TOPHAT, down_diagonal_kernel, iterations=1)

    # tophat_image = cv2.bitwise_or(cv2.bitwise_or(tophat_horizontal, tophat_vertical),
    #                                 cv2.bitwise_or(tophat_up_diag, tophat_down_diag))
    
    # dilate
    
    # Black hat filter to detect dark hairs
    

    # Binarize the image
    



    # Replace pixels of the mask
    # hairless_image = cv2.inpaint(hair_image, thresh_image, 6, cv2.INPAINT_TELEA)  

    intersection = np.logical_and(closed_image, hair_mask_image)
    union = np.logical_or(closed_image, hair_mask_image)
    IOU += np.sum(intersection) / np.sum(union)  

print(IOU / 50)