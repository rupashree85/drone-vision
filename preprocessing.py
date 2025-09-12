import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import json
import random
from sklearn.cluster import MiniBatchKMeans
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASSES_RETINANET, IMAGE_WIDTH, IMAGE_HEIGHT, MODEL_INPUT_SIZE_RETINANET, MODEL_INPUT_SIZE_ALEXNET

def create_directories():
    """Creates necessary directories for processed data."""
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'annotations_retinanet'), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'segmentation_masks_alexnet'), exist_ok=True)

def preprocess_image(image_path, target_size):
    """Loads and resizes an image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    img = cv2.resize(img, target_size)
    img = img / 255.0 # Normalize to [0, 1]
    return img

def process_retinanet_data(image_dir, annotation_path, output_image_dir, output_annotation_dir, target_size):
    """
    Processes images and annotations for RetinaNet.
    Includes data augmentation and normalization.
    Annotations are expected to be in COCO-like JSON format.
    """
    print(f"Processing RetinaNet data from {image_dir} and {annotation_path}...")
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    # Simplified example: iterate through images and save processed versions
    # In a real scenario, you'd integrate ImageDataGenerator or similar for augmentation
    for img_info in coco_data['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        original_width = img_info['width']
        original_height = img_info['height']
        image_path = os.path.join(image_dir, file_name)

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Skipping.")
            continue

        img = preprocess_image(image_path, target_size)
        
        # Save processed image
        processed_img_path = os.path.join(output_image_dir, file_name)
        cv2.imwrite(processed_img_path, cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        # Filter annotations for this image and adjust bounding boxes
        img_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        processed_annotations = []

        for ann in img_annotations:
            category_id = ann['category_id']
            # Map category_id to internal class index if necessary
            # For simplicity, assuming category_id directly maps to CLASSES_RETINANET index
            
            x_min, y_min, width, height = ann['bbox']
            
            # Scale bounding box coordinates
            x_min_scaled = (x_min / original_width) * target_size[0]
            y_min_scaled = (y_min / original_height) * target_size[1]
            width_scaled = (width / original_width) * target_size[0]
            height_scaled = (height / original_height) * target_size[1]

            processed_annotations.append({
                'image_id': img_id,
                'category_id': category_id,
                'bbox': [x_min_scaled, y_min_scaled, width_scaled, height_scaled]
            })
        
        # Save processed annotations (e.g., in a separate JSON per image or update master JSON)
        # For simplicity, here we just show the processing logic. A real implementation
        # would likely save a single new COCO-format JSON or a TFRecord.
        # This example will save per-image JSON for demonstration.
        base_name = os.path.splitext(file_name)[0]
        with open(os.path.join(output_annotation_dir, f"{base_name}.json"), 'w') as out_f:
            json.dump(processed_annotations, out_f)
    print("RetinaNet data processing complete.")


def process_alexnet_data(image_dir, mask_dir, output_image_dir, output_mask_dir, target_size):
    """
    Processes images and corresponding segmentation masks for AlexNet.
    Includes data augmentation (e.g., random flips) and normalization.
    Masks are expected as single-channel grayscale images where pixel values
    correspond to class IDs (e.g., 0 for non-water, 1 for water).
    """
    print(f"Processing AlexNet data from {image_dir} and {mask_dir}...")
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    for file_name in image_files:
        image_path = os.path.join(image_dir, file_name)
        mask_path = os.path.join(mask_dir, file_name) # Assuming mask has same name as image

        if not os.path.exists(mask_path):
            print(f"Warning: Mask {mask_path} not found for image {file_name}. Skipping.")
            continue

        img = preprocess_image(image_path, target_size)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize mask (e.g., if mask pixels are 0 and 255, convert to 0 and 1)
        mask = (mask > 0).astype(np.uint8) # Assuming water is non-zero, non-water is zero

        # Apply simple data augmentation (e.g., random horizontal flip)
        if random.random() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        # Save processed image and mask
        processed_img_path = os.path.join(output_image_dir, file_name)
        processed_mask_path = os.path.join(output_mask_dir, file_name)
        
        cv2.imwrite(processed_img_path, cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(processed_mask_path, (mask * 255).astype(np.uint8)) # Save as grayscale
    print("AlexNet data processing complete.")

def generate_adaptive_anchors(annotations_path, num_anchors=9):
    """
    Generates adaptive anchor boxes for RetinaNet using K-means clustering
    on bounding box widths and heights from the training dataset.
    Annotations are expected to be in COCO-like JSON format.
    """
    print("Generating adaptive anchors...")
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    bboxes = [] # Store (width, height) for clustering
    for ann in coco_data['annotations']:
        _, _, width, height = ann['bbox']
        bboxes.append([width, height])
    
    bboxes = np.array(bboxes)

    if len(bboxes) == 0:
        print("No bounding boxes found for anchor generation.")
        return []

    # Use MiniBatchKMeans for efficiency with large datasets
    kmeans = MiniBatchKMeans(n_clusters=num_anchors, random_state=0, n_init=10)
    kmeans.fit(bboxes)
    
    anchors = kmeans.cluster_centers_
    # Sort anchors by area for consistent ordering
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])] 
    
    print(f"Generated adaptive anchors (width, height): {anchors}")
    return anchors

if __name__ == "__main__":
    create_directories()

    # Example usage:
    # Assume raw data is structured as:
    # RAW_DATA_DIR/
    # ├── retinanet_images/ (e.g., from VisDrone, MS COCO)
    # ├── retinanet_annotations.json (COCO format)
    # ├── alexnet_images/ (e.g., from custom flood dataset)
    # └── alexnet_masks/ (segmentation masks for alexnet_images)
    
    # Process RetinaNet data
    retinanet_img_dir = os.path.join(RAW_DATA_DIR, 'retinanet_images')
    retinanet_ann_file = os.path.join(RAW_DATA_DIR, 'retinanet_annotations.json')
    if os.path.exists(retinanet_img_dir) and os.path.exists(retinanet_ann_file):
        process_retinanet_data(
            image_dir=retinanet_img_dir,
            annotation_path=retinanet_ann_file,
            output_image_dir=os.path.join(PROCESSED_DATA_DIR, 'images'),
            output_annotation_dir=os.path.join(PROCESSED_DATA_DIR, 'annotations_retinanet'),
            target_size=MODEL_INPUT_SIZE_RETINANET
        )
        # Generate and save adaptive anchors
        adaptive_anchors = generate_adaptive_anchors(retinanet_ann_file)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'retinanet_adaptive_anchors.npy'), adaptive_anchors)
    else:
        print(f"Skipping RetinaNet data processing. Missing directories/files: {retinanet_img_dir}, {retinanet_ann_file}")


    # Process AlexNet data
    alexnet_img_dir = os.path.join(RAW_DATA_DIR, 'alexnet_images')
    alexnet_mask_dir = os.path.join(RAW_DATA_DIR, 'alexnet_masks')
    if os.path.exists(alexnet_img_dir) and os.path.exists(alexnet_mask_dir):
        process_alexnet_data(
            image_dir=alexnet_img_dir,
            mask_dir=alexnet_mask_dir,
            output_image_dir=os.path.join(PROCESSED_DATA_DIR, 'images'),
            output_mask_dir=os.path.join(PROCESSED_DATA_DIR, 'segmentation_masks_alexnet'),
            target_size=MODEL_INPUT_SIZE_ALEXNET
        )
    else:
        print(f"Skipping AlexNet data processing. Missing directories/files: {alexnet_img_dir}, {alexnet_mask_dir}")
