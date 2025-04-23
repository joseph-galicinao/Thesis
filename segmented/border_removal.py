import os
import cv2
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

ROW = 2
COL = 3
SCALE = 0.5
AREA_THRESHOLD = 5
BRIGHTNESS_ERROR = 175
VARIANCE_ERROR = 32

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

top_left_kernel = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

# bottom_left_kernel = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
#                                [1, 1, 1, 0, 1, 1, 1, 1, 0],
#                                 [1, 1, 1, 0, 1, 1, 1, 0, 0],
#                                 [1, 1, 1, 1, 0, 0, 0, 0, 0],
#                                 [1, 1, 1, 1, 1, 0, 0, 0, 0],
#                                 [1, 1, 1, 1, 0, 1, 0, 0, 0],
#                                 [1, 1, 1, 0, 0, 0, 1, 0, 0],
#                                 [1, 1, 0, 0, 0, 0, 0, 1, 0],
#                                 [1, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.uint8)

# ORIGINAL_PATH = "./segmented/IOU_original_0.5/"
# SEGMENTED_PATH = "./segmented/IOU_segmented_0.5/"
ORIGINAL_PATH = "./segmented/original/"
SEGMENTED_PATH = "./segmented/segmentation/"
CSV_PATH = "./segmented/ISIC_classes.csv"

original_dir = os.listdir(ORIGINAL_PATH)
segmented_dir = os.listdir(SEGMENTED_PATH)

average_IOU = 0
num_images = 0

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

def calculate_IOU(mask, segmented_image):
    intersection = np.logical_and(segmented_image, mask)
    union = np.logical_or(segmented_image, mask)
    IOU = np.sum(intersection) / np.sum(union)

    return IOU

# Preprocessing functions
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

def remove_border(image):
    height, width, _ = image.shape
    L = lab_image[:,:,0]

    # Create Otsu Mask
    _, mask = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_mask = cv2.bitwise_not(mask)

    # Create Circle mask
    circle_mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    cv2.circle(circle_mask, center, max(height // 2, width // 2) - 3, 255, -1)
    circle_mask = cv2.bitwise_not(circle_mask)

    borderless_mask = cv2.bitwise_and(otsu_mask, circle_mask)

    return borderless_mask

def subsample(image, new_height, new_width):
    new_size = (new_width, new_height)
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    return resized_image

# Post processing functions
def get_lesion_components(lab_image, labeled_image, num_labels, areas, centroids):

    # Gets all the labels that are not the background
    valid_components = np.array([label for label in range(1, num_labels)])
    # print(f"Original components: {valid_components}")
    
    # Filters out components that have a small area
    area_mean = np.mean(areas)
    small_area_labels = np.argwhere(areas < area_mean).flatten()
    valid_components = np.delete(valid_components, small_area_labels)
    # print(f"Filtered by areas: {valid_components}")

    mask = np.isin(labeled_image, valid_components).astype(np.uint8) * 255

    plt.subplot(ROW, COL, 4)
    plt.title("Area Filter")
    plt.imshow(mask, cmap="gray")
    plt.axis('off')

    # Filter out components that are not centered
    height, width = labeled_image.shape
    center = np.array([width // 2, height // 2], dtype=np.int64)

    distances = [np.linalg.norm(centroids[label] - center)
        for label in valid_components]
    distance_mean = np.mean(distances)
    filtered_by_distance = np.argwhere(distances > distance_mean).flatten()
    valid_components = np.delete(valid_components, filtered_by_distance)
    # print(f"Filtered by centermost: {valid_components}")

    mask = np.isin(labeled_image, valid_components).astype(np.uint8) * 255

    plt.subplot(ROW, COL, 5)
    plt.title("Center Filter")
    plt.imshow(mask, cmap="gray")
    plt.axis('off')
    
    # Filter out components that are too light
    L = lab_image[:,:,0]
    brightness = np.array([np.mean(L[labels == label]) 
                           for label in valid_components])
    brightness_mean = np.mean(brightness)
    filtered_by_brightness = np.argwhere(brightness > brightness_mean).flatten()
    valid_components = np.delete(valid_components, filtered_by_brightness)
    
    mask = np.isin(labeled_image, valid_components).astype(np.uint8) * 255
    
    plt.subplot(ROW, COL, 6)
    plt.title("Brightness Filter")
    plt.imshow(mask, cmap="gray")
    plt.axis('off')

    # plt.show()

    return mask
    
# MAIN CODE
bad_image = 0
for original_file, segmented_file in zip(original_dir, segmented_dir):

    original_image = cv2.imread(ORIGINAL_PATH + original_file)
    original_image = subsample(original_image, 112, 150)
    
    plt.subplot(ROW, COL, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    segmented_image = cv2.imread(SEGMENTED_PATH + segmented_file)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    # segmented_image = subsample(segmented_image, SCALE)
    

    plt.subplot(ROW, COL, 2)
    plt.title("Segmented")
    plt.imshow(segmented_image, cmap="gray")
    plt.axis('off')

    # Remove hair
    hairless_image = remove_hair(original_image)

    # # Smooth image
    smoothed_image = cv2.GaussianBlur(hairless_image, (5, 5), 0)

    # Removes border
    lab_image = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2Lab)
    borderless_mask = remove_border(smoothed_image)

    # Perform EM Clustering on only the non-border pixels
    borderless_image = lab_image[borderless_mask != 255]
    borderless_image = borderless_image.reshape(-1, 3)

    scaler = StandardScaler()
    features = scaler.fit_transform(borderless_image)  # Each column has 0 mean, 1 std

    em = GaussianMixture(n_components=2)
    em.fit(features)
    labels = em.predict(features)
    
    # Assumes that the darker cluster is the skin lesion
    L_values = borderless_image[:, 0]  # Extract L channel (still in LAB space)
    darkness = np.array([np.mean(L_values[labels == i]) for i in range(em.n_components)])
    darkest_cluster = np.argmin(darkness)
    em_image = np.zeros_like(borderless_mask)  # Create a mask of the same shape as the original mask
    em_image[borderless_mask != 255] = (labels == darkest_cluster).astype(np.uint8) * 255

    plt.subplot(ROW, COL, 3)
    plt.title("EM")
    plt.imshow(em_image, cmap="gray")
    plt.axis('off')

    # Perform connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(em_image, connectivity=4)
    
    areas = np.array([stats[label][cv2.CC_STAT_AREA] for label in range(1, num_labels)])
    
    # Keep the components that are skin lesions
    mask = get_lesion_components(lab_image, labels, num_labels, areas, centroids)
    mask = subsample(mask, 450, 600)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    IOU = calculate_IOU(mask, segmented_image)

    # plt.show()

    if IOU < 0.5:
        bad_image += 1
        # cv2.imwrite(f"./segmented/IOU_segmented_0.5/{segmented_file}", segmented_image)
        # cv2.imwrite(f"./segmented/IOU_original_0.5/{original_file}", original_image)

    value = df.iloc[image]['dx']

    # # Update the amount of error according to the classification
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
    print(f"Image {image} done | IOU {IOU} | Num Error {bad_image}")

avg_bkl_IOU = bkl_IOU / bkl_num
avg_bcc_IOU = bcc_IOU / bcc_num
avg_akiec_IOU = akiec_IOU / akiec_num
avg_mel_IOU = mel_IOU / mel_num
avg_nv_IOU = nv_IOU / nv_num
avg_df_IOU = df_IOU / df_num
avg_vasc_IOU = vasc_IOU / vasc_num

print("EM Clustering Results")
print(f"BKL\t\t| Num: {bkl_num}\t\t| IOU: {avg_bkl_IOU}")
print(f"BCC\t\t| Num: {bcc_num}\t\t| IOU: {avg_bcc_IOU}")
print(f"AKIEC\t\t| Num: {akiec_num}\t\t| IOU: {avg_akiec_IOU}")
print(f"MEL\t\t| Num: {mel_num}\t\t| IOU: {avg_mel_IOU}")
print(f"NV\t\t| Num: {nv_num}\t\t| IOU: {avg_nv_IOU}")
print(f"DF\t\t| Num: {df_num}\t\t| IOU: {avg_df_IOU}")
print(f"VASC\t\t| Num: {vasc_num}\t\t| IOU: {avg_vasc_IOU}")