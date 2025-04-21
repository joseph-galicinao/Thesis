import os
import cv2
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

ORIGINAL_PATH = "./segmented/IOU_original_0.5/"
SEGMENTED_PATH = "./segmented/IOU_segmented_0.5/"
CSV_PATH = "./segmented/ISIC_classes.csv"

original_dir = os.listdir(ORIGINAL_PATH)
segmented_dir = os.listdir(SEGMENTED_PATH)

df = pd.read_csv(CSV_PATH)

nv_IOU = 0
nv_num = 0

mel_IOU = 0
mel_num = 0

bkl_IOU = 0
bkl_num = 0

bcc_IOU = 0
bcc_num = 0

akiec_IOU = 0
akiec_num = 0

df_IOU = 0
df_num = 0

vasc_IOU = 0
vasc_num = 0

image = 0

start_time = time.perf_counter()
for original_file, segmented_file in zip(original_dir, segmented_dir):
    # Load original image and get its metadata
    original_image = cv2.imread(ORIGINAL_PATH + original_file)
    height, width, _ = original_image.shape

    # Subsampled image
    new_size = (int(width), int(height))
    original_image = cv2.resize(original_image, new_size, interpolation=cv2.INTER_AREA)

    # Grayscale then perform Otsu thresholding
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, otsu_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_image = cv2.bitwise_not(otsu_image)

    # Perform Connected Components
    # num_labels, labels = cv2.connectedComponents(otsu_image, connectivity=4)
    num_labels, labels = cv2.connectedComponents(otsu_image, connectivity=8)
    _, counts = np.unique(labels, return_counts=True)
    counts[0] = 0 # Ignore the background

    # Assume that the largest connected component is the skin lesion
    largest_label = np.argmax(counts)
    otsu_image = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # Load segmented image and convert from BGR to grayscale
    segmented_image = cv2.imread(SEGMENTED_PATH + segmented_file)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    segmented_image = cv2.resize(segmented_image, new_size, interpolation=cv2.INTER_AREA)

    # Calculate the Intersection
    intersection = np.logical_and(otsu_image, segmented_image)
    union = np.logical_or(otsu_image, segmented_image)
    IOU = np.sum(intersection) / np.sum(union)

    # Determine skin lesion classification (based on the CSV file)
    value = df.iloc[image]['dx']

    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Otsu")
    plt.imshow(otsu_image, cmap="gray")
    plt.axis('off')


    plt.show()

    # Update the amount of error according to the classification
    if value == "bkl":
        bkl_IOU += IOU
        bkl_num += 1
    elif value == "bcc":
        bcc_IOU += IOU
        bcc_num += 1
    elif value == "akiec":
        akiec_IOU += IOU
        akiec_num += 1
    elif value == "mel":
        mel_IOU += IOU
        mel_num += 1
    elif value == "nv":
        nv_IOU += IOU
        nv_num += 1
    elif value == "df":
        df_IOU += IOU
        df_num += 1 
    elif value == "vasc":
        vasc_IOU += IOU
        vasc_num += 1

    image += 1
    print(f"Image {image} done")
end_time = time.perf_counter()

avg_bkl_IOU = bkl_IOU / bkl_num
avg_bcc_IOU = bcc_IOU / bcc_num
avg_akiec_IOU = akiec_IOU / akiec_num
avg_mel_IOU = mel_IOU / mel_num
avg_nv_IOU = nv_IOU / nv_num
avg_df_IOU = df_IOU / df_num
avg_vasc_IOU = vasc_IOU / vasc_num

print("Otsu Thresholding Results")
print(f"BKL\t\t| Num: {bkl_num}\t\t| Average Error: {avg_bkl_IOU}")
print(f"BCC\t\t| Num: {bcc_num}\t\t| Average Error: {avg_bcc_IOU}")
print(f"AKIEC\t\t| Num: {akiec_num}\t\t| Average Error: {avg_akiec_IOU}")
print(f"MEL\t\t| Num: {mel_num}\t\t| Average Error: {avg_mel_IOU}")
print(f"NV\t\t| Num: {nv_num}\t\t| Average Error: {avg_nv_IOU}")
print(f"DF\t\t| Num: {df_num}\t\t| Average Error: {avg_df_IOU}")
print(f"VASC\t\t| Num: {vasc_num}\t\t| Average Error: {avg_vasc_IOU}")

elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time} seconds")


