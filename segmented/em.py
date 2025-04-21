import os
import cv2
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

ROW = 1
COL = 5
SCALE = 0.25
AREA_THRESHOLD = 500
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

def calculate_IOU(mask, segmented_image):
    intersection = np.logical_and(segmented_image, mask)
    union = np.logical_or(segmented_image, mask)
    IOU = np.sum(intersection) / np.sum(union)

    return IOU

def get_distance(cluster):
    # Define the center
    x_center = cluster.shape[1] // 2
    y_center = cluster.shape[0] // 2

    # Get the coordinates of each pixel that is apart of the corresponding cluster
    y_coords, x_coords = np.where(cluster == 255)

    # Calculate the distance
    x_dif = np.subtract(x_coords, x_center)
    y_dif = np.subtract(y_coords, y_center)
    
    distance = np.sqrt(x_dif ** 2 + y_dif ** 2)
    avg_distance = distance.mean()

    return avg_distance

# def calculate_variation(cluster):

def get_lesion_label(original_image, em_image):
    cluster1 = np.where(em_image == 0, 255, 0).astype(np.uint8)
    distance1 = get_distance(cluster1)
    lab1 = original_image[cluster1 == 255]
    mean1 = np.mean(lab1, axis=0)

    cluster2 = np.where(em_image == 1, 255, 0).astype(np.uint8)
    distance2 = get_distance(cluster2)
    lab2 = original_image[cluster2 == 255]
    mean2 = np.mean(lab2, axis=0)

    cluster3 = np.where(em_image == 2, 255, 0).astype(np.uint8)
    distance3 = get_distance(cluster3)
    lab3 = original_image[cluster3 == 255]
    mean3 = np.mean(lab3, axis=0)

    # The maximum Euclidean distance is considered the border
    border_cluster = np.argmax([distance1, distance2, distance3])
    means = np.array([mean1, mean2, mean3])

    means = np.delete(means, border_cluster, axis=0)
    lab_distance = np.linalg.norm(means[0] - means[1])
    print(lab_distance)

    print(f"Border Cluster: {border_cluster}")

    # PSEUDO CODE:
    # Look at the variation in both labels combined

    borderless_mask = np.where(em_image != border_cluster, 255, 0)

    plt.subplot(ROW, COL, 4)
    plt.title("Borderless Mask")
    plt.imshow(borderless_mask, cmap="gray")
    plt.axis('off')

    # # original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    borderless_image = original_image[borderless_mask == 255]
    
    mean = np.mean(borderless_image, axis=0)
    # print(mean)

    # print(mean_image)
    #   If small variation --> combine labels into one
    #   Else the label with the darkest average is the skin lesion

    # The least bright is considered the skin lesion
    # all_centers = np.sum(em.means_, axis=1)
    # cluster_centers = np.delete(em.means_, border_cluster, axis=0)  # shape: (2, 3)

    # brightness = np.sum(cluster_centers, axis=1)  # simple sum of RGB

    # # Find error in brightnesses
    # dif_brightness = abs(brightness[0] - brightness[1])

    # #  If the error is small, then they are both the skin lesion
    # if sd < VARIANCE_ERROR:
    #     return borderless_mask
    # else:

    # # Else, get the darkest one
    # else:
    #     min_brightness = np.min(brightness)
    #     skin_lesion_label = np.argwhere(all_centers == min_brightness)

    #     segmented_image = np.where(em_image == skin_lesion_label[0][0], 255, 0)
    #     return segmented_image

def final_mask(image):
    num_labels, labels = cv2.connectedComponents(image, connectivity=4)
    _, areas = np.unique(labels, return_counts=True)
    areas[0] = 0 # Ignore the background

    # Filter labels where the area is greater than the area threshold
    area_filtered_labels = np.argwhere(areas > AREA_THRESHOLD)

    avg_distance = []
    for label in range(len(area_filtered_labels)):
        component = np.where(labels == area_filtered_labels[label][0], 255, 0)
        distance = get_distance(component)
        avg_distance.append(distance)
    
    best_distance_index = np.argmin(avg_distance)
    skin_lesion_label = area_filtered_labels[best_distance_index][0]

    mask = np.where(labels == skin_lesion_label, 255, 0)



    # print(distances)

    # # Filter labels that are near the center


        
    # Filter labels that are the darkest

    return mask
       
# MAIN CODE
for original_file, segmented_file in zip(original_dir, segmented_dir):
    # Load original image and get its metadata
    original_image = cv2.imread(ORIGINAL_PATH + original_file)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab)

    plt.subplot(ROW, COL, 1)
    plt.title("Original")
    plt.imshow(original_image, cmap="gray")
    plt.axis('off')

    # Load segmented image and convert from BGR to grayscale
    segmented_image = cv2.imread(SEGMENTED_PATH + segmented_file)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    plt.subplot(ROW, COL, 2)
    plt.title("Mask")
    plt.imshow(segmented_image, cmap="gray")
    plt.axis('off')

    # Subsampled images
    new_size = (int(original_image.shape[1] * SCALE), int(original_image.shape[0] * SCALE))
    resized_image = cv2.resize(original_image, new_size, interpolation=cv2.INTER_AREA)
    height, width, channels = resized_image.shape
    resized_segmented_image = cv2.resize(segmented_image, new_size, interpolation=cv2.INTER_AREA)

    # Remove hair
    hairless_image = remove_hair(resized_image)

    # Smooth image
    blurred_image = cv2.medianBlur(hairless_image, 7, 0)
    # blurred_image = cv2.GaussianBlur(hairless_image, (3, 3), 0)

    # Perform EM Clustering
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten all components
    L = blurred_image[:, :, 0].flatten()
    a = blurred_image[:, :, 1].flatten()
    b = blurred_image[:, :, 2].flatten()
    # x = x_coords.flatten()
    # y = y_coords.flatten()

    features = np.stack((L, a, b), axis=1)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)  # Each column has 0 mean, 1 std

    # reshaped_image = blurred_image.reshape(height * width, channels)
    em = GaussianMixture(n_components=2)
    em.fit(features)
    labels = em.predict(features)
    em_image = labels.reshape(height, width).astype(np.uint8)

    L_means = em.means_[:, 0]  # Lightness values
    # print(L_means)

    darkest_cluster = np.argmin(L_means)
    # print(darkest_cluster)

    plt.subplot(ROW, COL, 3)
    plt.title("EM")
    plt.imshow(em_image, cmap="gray")
    plt.axis('off')

    dark_mask = (em_image == darkest_cluster).astype(np.uint8) * 255

    plt.subplot(ROW, COL, 4)
    plt.title("EM")
    plt.imshow(dark_mask, cmap="gray")
    plt.axis('off')

    # Obtain the label that corresponds to the skin lesion
    # skin_lesion_label = get_lesion_label(blurred_image, em_image)
    # em_image = np.where(em_image == skin_lesion_label, 255, 0).astype(np.uint8)

    # conn_image = final_mask(em_image)

    # plt.subplot(ROW, COL, 4)
    # plt.title("EM")
    # plt.imshow(skin_lesion_label, cmap="gray")
    # plt.axis('off')

    num_labels, labels = cv2.connectedComponents(dark_mask, connectivity=4)
    _, areas = np.unique(labels, return_counts=True)
    areas[0] = 0 # Ignore the background
    largest_component = np.argmax(areas)
    mask = np.where(labels == largest_component, 255, 0).astype(np.uint8)

    IOU = calculate_IOU(mask, resized_segmented_image)

    # plt.show()

    # Determine skin lesion classification (based on the CSV file)
    value = df.iloc[image]['dx']

    if IOU < 0.5:
        cv2.imwrite(f"./segmented/IOU_segmented_0.5/{segmented_file}", segmented_image)

        original_image = cv2.cvtColor(original_image, cv2.COLOR_Lab2BGR)
        cv2.imwrite(f"./segmented/IOU_original_0.5/{original_file}", original_image)

    

    # plt.subplot(ROW, COL, 7)
    # plt.title("Final Mask")
    # plt.imshow(conn_image, cmap="gray")
    # plt.axis('off')

    # plt.show()

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
print(f"BKL\t\t| Num: {bkl_num}\t\t| IOU: {avg_bkl_IOU}")
print(f"BCC\t\t| Num: {bcc_num}\t\t| IOU: {avg_bcc_IOU}")
print(f"AKIEC\t\t| Num: {akiec_num}\t\t| IOU: {avg_akiec_IOU}")
print(f"MEL\t\t| Num: {mel_num}\t\t| IOU: {avg_mel_IOU}")
print(f"NV\t\t| Num: {nv_num}\t\t| IOU: {avg_nv_IOU}")
print(f"DF\t\t| Num: {df_num}\t\t| IOU: {avg_df_IOU}")
print(f"VASC\t\t| Num: {vasc_num}\t\t| IOU: {avg_vasc_IOU}")