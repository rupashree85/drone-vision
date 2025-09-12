import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..')

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Model paths
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RETINANET_MODELS_DIR = os.path.join(MODELS_DIR, 'retinanet')
ALEXNET_MODELS_DIR = os.path.join(MODELS_DIR, 'alexnet')
OPTIMIZED_MODELS_DIR = os.path.join(MODELS_DIR, 'optimized')

# Pretrained weights (e.g., ImageNet for ResNet50 backbone)
IMAGENET_WEIGHTS_PATH = 'path/to/imagenet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Image dimensions
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 480
MODEL_INPUT_SIZE_RETINANET = (448, 448)
MODEL_INPUT_SIZE_ALEXNET = (224, 224)

# Training parameters
BATCH_SIZE = 8
EPOCHS_RETINANET = 50
EPOCHS_ALEXNET = 30
LEARNING_RATE = 1e-4

# Object Detection Classes for RetinaNet
CLASSES_RETINANET = ['background', 'person', 'debris', 'fallen_tree', 'vehicle']
NUM_CLASSES_RETINANET = len(CLASSES_RETINANET)

# Segmentation Classes for AlexNet
CLASSES_ALEXNET = ['non_water', 'water']
NUM_CLASSES_ALEXNET = len(CLASSES_ALEXNET)

# RetinaNet specific parameters
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0
NMS_IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.05 # Minimum confidence for a detection

# AlexNet specific parameters
ALEXNET_OUTPUT_CHANNELS = NUM_CLASSES_ALEXNET # For segmentation

# Raspberry Pi deployment parameters
TFLITE_MODEL_RETINANET = os.path.join(OPTIMIZED_MODELS_DIR, 'retinanet_quantized.tflite')
TFLITE_MODEL_ALEXNET = os.path.join(OPTIMIZED_MODELS_DIR, 'alexnet_quantized.tflite')

# Docker setup
DOCKER_IMAGE_NAME = "drone_ml_inference"
DOCKERFILE_PATH = os.path.join(PROJECT_ROOT, 'Dockerfile.inference')
