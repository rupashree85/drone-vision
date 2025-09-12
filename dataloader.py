import os
import numpy as np
import tensorflow as tf
import cv2
import json
from config import PROCESSED_DATA_DIR, CLASSES_RETINANET, CLASSES_ALEXNET, MODEL_INPUT_SIZE_RETINANET, MODEL_INPUT_SIZE_ALEXNET, BATCH_SIZE

class DataLoader:
    def __init__(self, mode='train'):
        self.mode = mode
        self.image_dir = os.path.join(PROCESSED_DATA_DIR, 'images')

    def _load_image(self, image_path, target_size):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0 # Normalize
        return img

class RetinaNetDataLoader(DataLoader):
    def __init__(self, mode='train'):
        super().__init__(mode)
        self.annotation_dir = os.path.join(PROCESSED_DATA_DIR, 'annotations_retinanet')
        self.image_files = sorted(os.listdir(self.image_dir)) # Example, filter by available annotations
        # For simplicity, let's assume image_files correspond to processed_annotations
        # In a real scenario, you'd load a single COCO JSON and parse it.
        
        # Dummy data loading for illustration:
        self.data = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            ann_file = os.path.join(self.annotation_dir, f"{base_name}.json")
            if os.path.exists(ann_file):
                with open(ann_file, 'r') as f:
                    annotations = json.load(f)
                self.data.append((os.path.join(self.image_dir, img_file), annotations))

    def _parse_annotation(self, annotations, image_width, image_height):
        """Parses annotations and converts to model-ready format (boxes, labels)."""
        boxes = []
        labels = []
        for ann in annotations:
            x_min, y_min, width, height = ann['bbox']
            # Convert [x_min, y_min, w, h] to [x1, y1, x2, y2]
            boxes.append([x_min, y_min, x_min + width, y_min + height])
            labels.append(ann['category_id']) # Assuming category_id maps directly to class index
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int32)

    def generate_batch(self):
        """Generates batches of data for RetinaNet training."""
        while True:
            batch_images = []
            batch_boxes = []
            batch_labels = []

            # Shuffle data for each epoch
            random_indices = np.arange(len(self.data))
            np.random.shuffle(random_indices)

            for i in random_indices:
                image_path, annotations = self.data[i]
                img = self._load_image(image_path, MODEL_INPUT_SIZE_RETINANET)
                boxes, labels = self._parse_annotation(annotations, MODEL_INPUT_SIZE_RETINANET[0], MODEL_INPUT_SIZE_RETINANET[1])

                if len(boxes) == 0:
                    continue # Skip images with no annotations

                batch_images.append(img)
                batch_boxes.append(boxes)
                batch_labels.append(labels)

                if len(batch_images) == BATCH_SIZE:
                    # Pad boxes and labels to ensure consistent batch shapes
                    max_boxes = max([len(b) for b in batch_boxes])
                    padded_boxes = np.zeros((BATCH_SIZE, max_boxes, 4), dtype=np.float32)
                    padded_labels = np.zeros((BATCH_SIZE, max_boxes), dtype=np.int32) - 1 # Use -1 for padding label

                    for idx, (b, l) in enumerate(zip(batch_boxes, batch_labels)):
                        padded_boxes[idx, :len(b)] = b
                        padded_labels[idx, :len(l)] = l
                    
                    yield tf.constant(np.array(batch_images), dtype=tf.float32), \
                          {'boxes': tf.constant(padded_boxes, dtype=tf.float32), 
                           'labels': tf.constant(padded_labels, dtype=tf.int32)}
                    
                    batch_images = []
                    batch_boxes = []
                    batch_labels = []

class AlexNetDataLoader(DataLoader):
    def __init__(self, mode='train'):
        super().__init__(mode)
        self.mask_dir = os.path.join(PROCESSED_DATA_DIR, 'segmentation_masks_alexnet')
        self.image_files = sorted(os.listdir(self.image_dir)) # Example, filter by available masks

        # Dummy data loading for illustration:
        self.data = []
        for img_file in self.image_files:
            mask_file = os.path.join(self.mask_dir, img_file) # Assuming mask has same name
            if os.path.exists(mask_file):
                self.data.append((os.path.join(self.image_dir, img_file), mask_file))

    def _load_mask(self, mask_path, target_size):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) # Binary mask (0 or 1)
        # One-hot encode the mask for categorical crossentropy
        mask = tf.keras.utils.to_categorical(mask, num_classes=len(CLASSES_ALEXNET))
        return mask

    def generate_batch(self):
        """Generates batches of data for AlexNet training."""
        while True:
            batch_images = []
            batch_masks = []

            random_indices = np.arange(len(self.data))
            np.random.shuffle(random_indices)

            for i in random_indices:
                image_path, mask_path = self.data[i]
                img = self._load_image(image_path, MODEL_INPUT_SIZE_ALEXNET)
                mask = self._load_mask(mask_path, MODEL_INPUT_SIZE_ALEXNET)

                batch_images.append(img)
                batch_masks.append(mask)

                if len(batch_images) == BATCH_SIZE:
                    yield tf.constant(np.array(batch_images), dtype=tf.float32), \
                          tf.constant(np.array(batch_masks), dtype=tf.float32)
                    
                    batch_images = []
                    batch_masks = []
