import json
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

def calculate_overlap(image_size, patch_size=256, num_patches=2):
    """Calculate the appropriate overlap."""
    overlap = (patch_size * num_patches - image_size) / (num_patches - 1)
    overlap_rate = overlap / patch_size
    return overlap_rate

def downsample(image, factor=2):
    """Downsample the image by a given factor."""
    width, height = image.size
    new_width = width // factor
    new_height = height // factor
    return image.resize((new_width, new_height), Image.BILINEAR)

def extract_patches(image, patch_size=(256, 256), overlap_w=0.5, overlap_h=0.5):
    """Extract patches with overlap from the image."""
    width, height = image.size
    patch_w, patch_h = patch_size
    stride_w = int(patch_w * (1 - overlap_w))
    stride_h = int(patch_h * (1 - overlap_h))

    patches = []
    for i in range(0, height - patch_h + 1, stride_h):
        for j in range(0, width - patch_w + 1, stride_w):
            patch = image.crop((j, i, j + patch_w, i + patch_h))
            patches.append(np.array(patch))

    return patches

def process_images(data_path, img_names, output_dir, num_patches_w=6, num_patches_h=6):
    """Process images and save patches."""
    labels = {"labels": []}
    os.makedirs(output_dir, exist_ok=True)

    for img_name in tqdm(img_names):
        if not img_name.endswith(".tif"):
            continue
        img_path = os.path.join(data_path, img_name)
        image = Image.open(img_path)  # Load the image
        downsampled_image = downsample(image, factor=2)  # Downsample the image

        # Calculate the best overlap
        overlap_w = calculate_overlap(downsampled_image.size[0], patch_size=256, num_patches=num_patches_w)
        overlap_h = calculate_overlap(downsampled_image.size[1], patch_size=256, num_patches=num_patches_h)

        patches = extract_patches(downsampled_image, patch_size=(256, 256), overlap_w=overlap_w, overlap_h=overlap_h)  # Extract patches

        for i, patch in enumerate(patches):
            patch_img = Image.fromarray(patch)
            patch_filename = f"{img_name.split('.')[0]}_{i}.png"
            patch_img.save(os.path.join(output_dir, patch_filename))
            labels["labels"].append([patch_filename, f"{img_name.split('.')[0]}_{i}"])

    # Save metadata
    with open(os.path.join(output_dir, "dataset.json"), "w") as f:
        json.dump(labels, f)

def create_train_test_data(data_path, save_dir, train_ratio, num_patches_w=6, num_patches_h=6):
    files = []
    for f in os.listdir(data_path):
        if f.endswith(".tif"):
            files.append(f)
    train_size = int(len(files) * train_ratio)
    test_size = len(files) - train_size
    train_imgs, test_imgs = train_test_split(files, train_size=train_size, test_size=test_size)
    train_dir = os.path.join(save_dir, 'train')
    test_dir = os.path.join(save_dir, 'test')

    process_images(data_path, train_imgs, train_dir, num_patches_w, num_patches_h)
    process_images(data_path, test_imgs, test_dir, num_patches_w, num_patches_h)


train_ratio = 0.8
pre_event_path = 'Germany_Training_Public/PRE-event'
data_path = os.path.join('data/raw_data', pre_event_path)
output_path = os.path.join('data/patches', pre_event_path)


# Process the images to extract patches
create_train_test_data(data_path, output_path, train_ratio)