# Autonomous Drone ML Subsystem

This repository contains the machine learning components for the Autonomous Flood & Disaster Response Drone System. It includes data preprocessing pipelines, custom implementations and training scripts for RetinaNet (object detection) and AlexNet (flood segmentation), model optimization for edge deployment, and inference scripts for real-time operation on a Raspberry Pi.

## Directory Structure

- `data/`: Stores raw and processed datasets.
- `models/`: Contains model architectures and trained weights (both full and optimized TFLite versions).
- `src/`: Core Python scripts for configuration, preprocessing, data loading, training, optimization, and inference.
- `scripts/`: Shell scripts for setting up the Raspberry Pi and deploying the Docker container.
- `notebooks/`: Jupyter notebooks for data exploration and model evaluation.

## Key Features

- **RetinaNet**: Object detection for `person`, `debris`, `fallen_tree`, `vehicle`.
- **AlexNet**: Semantic segmentation for `water` vs `non-water` classification.
- **Focal Loss**: Implemented for RetinaNet to handle class imbalance.
- **Model Quantization**: TensorFlow Lite optimization for Raspberry Pi deployment.
- **Adaptive Anchors**: K-means based anchor generation for RetinaNet.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/autonomous-drone-ml.git
    cd autonomous-drone-ml
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare Datasets**: Download and place your VisDrone, MS COCO, and custom flood segmentation datasets into `data/raw/`. Run `src/preprocessing.py` to generate processed data.

## Usage

- **Training**: See `src/train_retinanet.py` and `src/train_alexnet.py`.
- **Optimization**: Use `src/model_optimization.py` to convert models to TFLite.
- **Inference (Raspberry Pi)**: Refer to `src/inference.py` and `scripts/deploy_docker.sh`.
