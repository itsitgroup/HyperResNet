import os
import urllib.request
import zipfile
import numpy as np
import cv2

def download_and_extract_data(url, dest_folder, zip_name='complete_ms_data.zip'):
    """
    Download and extract data if it doesn't already exist.
    
    Args:
        url (str): URL to download the zip file.
        dest_folder (str): Destination folder to extract the contents.
        zip_name (str): Name of the zip file.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        
    zip_path = os.path.join(dest_folder, zip_name)
    
    if not os.path.exists(zip_path):
        print("Downloading data...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download completed.")

        print("Extracting data...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        print("Extraction completed.")
    else:
        print("Data already downloaded and extracted.")

def load_and_preprocess_images(root_folder, target_size):
    cropped_images = []
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for subdir2 in os.listdir(subdir_path):
            subdir2_path = os.path.join(subdir_path, subdir2)
            if not os.path.isdir(subdir2_path):
                continue

            stacked_img = np.empty([512, 512, 31])
            for i in range(1, 32):
                img_path = os.path.join(subdir2_path, f"{subdir2}_{i:02d}.png")
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                stacked_img[:, :, i-1] = img

            # Extract crops
            for h in range(0, 512 - target_size + 1, target_size // 2):
                for w in range(0, 512 - target_size + 1, target_size // 2):
                    cropped_img = stacked_img[h:h + target_size, w:w + target_size, :]
                    cropped_images.append(cropped_img)

    return np.array(cropped_images)

def generate_low_res_hsi_and_high_res_rgb(cropped_images):
    LowResHSI = []
    HiResRGB = []

    for img in cropped_images:
        # Low-resolution HSI
        low_res_img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        LowResHSI.append(low_res_img)

        # High-resolution RGB
        high_res_img = np.zeros((64, 64, 3))
        for i in range(3):
            high_res_img[:, :, i] = np.mean(img[:, :, i*10:(i+1)*10], axis=2)
        HiResRGB.append(high_res_img)

    return np.array(LowResHSI), np.array(HiResRGB)

def normalize_data(X_low_res_hsi_train, X_low_res_hsi_test, X_hi_res_rgb_train, X_hi_res_rgb_test, y_train, y_test):
    max_pixel_value = 255
    X_low_res_hsi_train = X_low_res_hsi_train / max_pixel_value
    X_low_res_hsi_test = X_low_res_hsi_test / max_pixel_value
    X_hi_res_rgb_train = X_hi_res_rgb_train / max_pixel_value
    X_hi_res_rgb_test = X_hi_res_rgb_test / max_pixel_value
    y_train = y_train / max_pixel_value
    y_test = y_test / max_pixel_value

    return X_low_res_hsi_train, X_low_res_hsi_test, X_hi_res_rgb_train, X_hi_res_rgb_test, y_train, y_test
