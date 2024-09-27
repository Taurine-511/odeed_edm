import json
import os
from tqdm import tqdm
from PIL import Image
import numpy as np


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

def process_images(data_path, output_dir, num_patches_w=5, num_patches_h=6):
    """Process images and save patches."""
    labels = {"labels": []}
    os.makedirs(output_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(data_path)):
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


pre_event_path = 'Germany_Training_Public/PRE-event'
data_path = os.path.join('data/raw_data', pre_event_path)
output_path = os.path.join('data/patches', pre_event_path)


# Process the images to extract patches
process_images(data_path, output_path)