import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

horizontal_kernel = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8)
vertical_kernel = np.array([[0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [0]], dtype=np.uint8)
down_diagonal_kernel = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
up_diagonal_kernel = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

ORIGINAL_PATH = "./segmented/original/"
SEGMENTED_PATH = "./segmented/segmentation/"
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

def remove_hair(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Edge Detection
    gaussian_image = cv2.GaussianBlur(grayscale_image, (7, 7), 0)
    canny_image = cv2.Canny(gaussian_image, threshold1=50, threshold2=150)

    close_horizontal = cv2.morphologyEx(canny_image, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)
    close_vertical = cv2.morphologyEx(canny_image, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    close_up_diag = cv2.morphologyEx(canny_image, cv2.MORPH_CLOSE, up_diagonal_kernel, iterations=1)
    close_down_diag = cv2.morphologyEx(canny_image, cv2.MORPH_CLOSE, down_diagonal_kernel, iterations=1)

    close_image = cv2.bitwise_or(cv2.bitwise_or(close_horizontal, close_vertical),
                                    cv2.bitwise_or(close_up_diag, close_down_diag))
    
    final_image = cv2.inpaint(image, close_image, 5, cv2.INPAINT_TELEA)
    return final_image

for original_file, segmented_file in zip(original_dir, segmented_dir):
    # Load original image and get its metadata
    original_image = cv2.imread(ORIGINAL_PATH + original_file)

    # Subsampled image
    new_size = (int(original_image.shape[1]), int(original_image.shape[0]))
    # new_size = (int(original_image.shape[1] * 0.5), int(original_image.shape[0] * 0.5))
    # new_size = (int(original_image.shape[1] * 0.25), int(original_image.shape[0] * 0.25))

    original_image = cv2.resize(original_image, new_size, interpolation=cv2.INTER_AREA)

    height, width, channels = original_image.shape

    # Perform K-means Clustering
    color_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    color_image = remove_hair(color_image)

    # color_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    # color_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab)
    color_image = color_image.reshape(height * width, channels)
    kmeans = KMeans(n_clusters=2, n_init='auto')
    trained = kmeans.fit(color_image)
    labels = trained.labels_
    kmeans_image = labels.reshape(height, width)

    # Determine which pixels are foreground and background
    cluster_centers = kmeans.cluster_centers_  # shape: (2, 3)
    brightness = np.sum(cluster_centers, axis=1)  # simple sum of RGB
    foreground_label = np.argmin(brightness)

    # Create binary image: foreground is white (255), background is black (0)
    kmeans_image = np.where(kmeans_image == foreground_label, 255, 0).astype(np.uint8)

    # Perform Connected Components
    # num_labels, labels = cv2.connectedComponents(kmeans_image, connectivity=4)
    # num_labels, labels = cv2.connectedComponents(kmeans_image, connectivity=8)
    # _, counts = np.unique(labels, return_counts=True)
    # counts[0] = 0 # Ignore the background

    # # Assume that the largest connected component is the skin lesion
    # largest_label = np.argmax(counts)
    # kmeans_image = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # Load segmented image and convert from BGR to grayscale
    segmented_image = cv2.imread(SEGMENTED_PATH + segmented_file)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Subsampled segmented image
    segmented_image = cv2.resize(segmented_image, new_size, interpolation=cv2.INTER_AREA)

    # Calculate the Intersection
    intersection = np.logical_and(kmeans_image, segmented_image)
    union = np.logical_or(kmeans_image, segmented_image)
    IOU = np.sum(intersection) / np.sum(union)

    # Determine skin lesion classification (based on the CSV file)
    value = df.iloc[image]['dx']

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
    print(f"Image {image} done | IOU {IOU}")

avg_bkl_IOU = bkl_IOU / bkl_num
avg_bcc_IOU = bcc_IOU / bcc_num
avg_akiec_IOU = akiec_IOU / akiec_num
avg_mel_IOU = mel_IOU / mel_num
avg_nv_IOU = nv_IOU / nv_num
avg_df_IOU = df_IOU / df_num
avg_vasc_IOU = vasc_IOU / vasc_num

print("KMeans Clustering Results")
print(f"BKL\t\t| Num: {bkl_num}\t\t| Average Error: {avg_bkl_IOU}")
print(f"BCC\t\t| Num: {bcc_num}\t\t| Average Error: {avg_bcc_IOU}")
print(f"AKIEC\t\t| Num: {akiec_num}\t\t| Average Error: {avg_akiec_IOU}")
print(f"MEL\t\t| Num: {mel_num}\t\t| Average Error: {avg_mel_IOU}")
print(f"NV\t\t| Num: {nv_num}\t\t| Average Error: {avg_nv_IOU}")
print(f"DF\t\t| Num: {df_num}\t\t| Average Error: {avg_df_IOU}")
print(f"VASC\t\t| Num: {vasc_num}\t\t| Average Error: {avg_vasc_IOU}")