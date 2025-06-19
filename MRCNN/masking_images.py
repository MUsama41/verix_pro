import os
import json
import cv2
import numpy as np
import difflib
from pathlib import Path

# Paths
IMAGE_DIR = "/home/dev/my_projects/Mask RCNN/verix-pro-v2/silt_fence-20250121T202609Z-001/train/images"
MASK_DIR = "/home/dev/my_projects/Mask RCNN/verix-pro-v2/silt_fence-20250121T202609Z-001/train/masked_4"
ANNOTATIONS_FILE = "/home/dev/my_projects/Mask RCNN/verix-pro-v2/silt_fence-20250121T202609Z-001/train/images/silt_fence_annotations.json"

# Ensure the mask directory exists
os.makedirs(MASK_DIR, exist_ok=True)

# Load JSON annotations
with open(ANNOTATIONS_FILE, "r") as f:
    annotations = json.load(f)

# Class label mapping
CLASS_MAPPING = {
    "silt_fence_type_1": 1,
    "silt_fence_type_2": 2,
    "silt_fence_type_3": 3,
    "silt_fence_type_4": 4
}

# Normalize available image filenames
available_images = {f.lower().strip().replace(" ", "_"): f for f in os.listdir(IMAGE_DIR)}

# Function to find closest matching filename
def find_closest_filename(target_name, available_files):
    normalized_target = target_name.lower().strip().replace(" ", "_")
    possible_matches = difflib.get_close_matches(normalized_target, available_files.keys(), n=1, cutoff=0.6)
    return available_files[possible_matches[0]] if possible_matches else None

# Process each image annotation
for key, data in annotations.items():
    filename = data.get("filename", "").strip()

    if not filename:
        print(f"Skipping entry with missing filename: {key}")
        continue

    # Normalize base name
    base_name = Path(filename).stem.lower().strip().replace(" ", "_")

    # Find closest matching image file
    found_image = find_closest_filename(base_name, available_images)

    if not found_image:
        print(f"⚠️ Image not found for annotation: {filename}")
        continue

    image_path = os.path.join(IMAGE_DIR, found_image)

    # Load the image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        continue

    height, width = image.shape[:2]

    # Create an empty mask (black background)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Process each region (polygon)
    for region in data.get("regions", []):
        shape_attrs = region.get("shape_attributes", {})
        all_points_x = shape_attrs.get("all_points_x", [])
        all_points_y = shape_attrs.get("all_points_y", [])
        class_name = region.get("region_attributes", {}).get("item_name")

        if class_name not in CLASS_MAPPING:
            print(f"⚠️ Skipping unknown class: {class_name}")
            continue

        class_value = CLASS_MAPPING[class_name]  # Assign class label

        if len(all_points_x) > 2 and len(all_points_y) > 2:
            # Convert polygon points to NumPy array
            points = np.array([list(zip(all_points_x, all_points_y))], dtype=np.int32)

            # **DEBUG STEP: Draw the polygon boundary**
            cv2.polylines(mask, [points], isClosed=True, color=255, thickness=2)

            # **Fill the polygon with the corresponding class label**
            cv2.fillPoly(mask, [points], class_value)

    # Save the mask
    mask_filename = Path(found_image).stem + "_mask.png"
    mask_path = os.path.join(MASK_DIR, mask_filename)
    
    # Save original and scaled versions for debugging
    #cv2.imwrite(mask_path, mask)
    cv2.imwrite(mask_path.replace(".png", "_debug.png"), mask * 50)  # Scaled for visibility
    
    print(f"✅ Mask saved: {mask_path}")

print("🎉 Multi-Class Mask Generation Completed.")
