import os
import numpy as np
from Training_utils import CustomDataset, CustomConfig
from mrcnn.model import MaskRCNN
from mrcnn.utils import download_trained_weights
from Training_utils import load_datasets

def main():
    # Paths
    # dataset_path = "/home/dev/my_projects/verix-pro-v2/silt_fence-20250121T202609Z-001"
    dataset_path = "data"

    weights_path = "data/weights/mask_rcnn_object_0005.h5"
    DEFAULT_LOGS_DIR = os.path.join(os.getcwd(), "logs")

    FileType_path = "data/JSON_files/types.json"
    annotation_json_path = "rock_berm_annotations.json"

    # Load training and validation datasets using the function
    dataset_train, dataset_val = load_datasets(dataset_path,FileType_path,annotation_json_path)


    # Configuration
    config = CustomConfig()
    #config.NUM_CLASSES = 1 + count  # Background + number of classes
    print("---------------------------------------------------------------------------num of classes : ",config.NUM_CLASSES )

    # Initialize the model
    model = MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

    # Download weights if not available
    if not os.path.exists(weights_path):
        download_trained_weights(weights_path)

    # Load the pre-trained weights
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    # Train the model
    print("Training network heads...")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='heads')

if __name__ == "__main__":
    main()
