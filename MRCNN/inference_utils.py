from mrcnn.config import Config  # Correct import
from pdf2image import convert_from_path
from PIL import Image
import os
from google.oauth2 import service_account

#import mrcnn.model as modellib
from google.cloud import storage
import matplotlib.pyplot as plt
import warnings

import cv2
from mrcnn.visualize import display_instances
from flask import jsonify, request
import requests
import json
import tempfile
import tensorflow as tf
import mrcnn
import numpy as np
from mrcnn import model as modellib, utils
import mrcnn.model as modellib

warnings.filterwarnings('ignore')



from urllib.parse import unquote, urlparse



num_classes = 0 # Background + 5 object classes

class CustomConfig(mrcnn.config.Config):
    NAME = "object"
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 2
    DETECTION_MIN_CONFIDENCE = 0.9

    def __init__(self, num_classes):
        self.NUM_CLASSES = 1 + num_classes  # Background + dynamic classes
        super().__init__()  # Call parent constructor



def resize_image(image, max_dim=7200):
    height, width = image.shape[:2]
    scale = min(max_dim / max(height, width), 1.0)  # Ensure we only downscale
    new_size = (int(width * scale), int(height * scale))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image



def create_pdf_from_images(image_paths, output_pdf_path):  #
    valid_image_paths = [path for path in image_paths if os.path.isfile(path)]
    
    if not valid_image_paths:
        print("******No valid image files found.")
        return
    
    if not output_pdf_path.endswith('.pdf'):
        output_pdf_path += '.pdf'
    
    images = [Image.open(image_path) for image_path in valid_image_paths]
    images[0].save(output_pdf_path, save_all=True, append_images=images[1:], resolution=100.0, quality=95)
    print(f"#######PDF saved at {output_pdf_path}")

def pdf_to_jpeg(pdf_path, output_folder):
    images = convert_from_path(pdf_path, dpi=72)
    image_paths = []
    
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i+1}.jpeg")
        image.save(image_path, 'JPEG')
        image_paths.append(image_path)
        print(f"###########Image saved at {image_path}")
    
    return image_paths

def load_model_for_segment(segment_name, model_dir):  #
    # Construct the model path based on the segment type
    print("segment name : ",segment_name)
    print("segment model dir : ",model_dir)
    # Root directory of the project
    ROOT_DIR = os.getcwd()
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    model_path = os.path.join(model_dir, f"{segment_name}.h5")
    
    if not os.path.exists(model_path):
        print(f"********Model file for {segment_name} not found!")
        return None
    
    if segment_name == "silt_fence":
        num_classes = 4
    elif segment_name == "rock_berms":
        num_classes = 6

    #config = CustomConfig()  # Use your custom config or a relevant one for detection
    # Example usage:
    config = CustomConfig(num_classes)
    print(config.NUM_CLASSES)  # Output: 5
    print("###############classes are : ",config.NUM_CLASSES)

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    #model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)
    print(f"###########Loading model for {segment_name} from {model_path}")
    model.load_weights(model_path, by_name=True)
    
    return model





# Initialize the GCP client and bucket
def initialize_gcp_client():
    # Path to your service account JSON file
    key_path = '/home/ubuntu/maskrcnn/verix-pro-v2/data/JSON_files/GCP_json.json'  # Replace with your credentials file path

    # Authenticate using the key
    credentials = service_account.Credentials.from_service_account_file(key_path)

    # Initialize the Google Cloud Storage client with the provided credentials
    client = storage.Client(credentials=credentials, project='level-sol-440022-t2')  # Replace with your project ID

    # Access the bucket
    bucket = client.get_bucket('weights_maskrcnn')  # Replace with your bucket name
    return bucket  # Replace with your bucket name

# Function to upload PDF to GCP bucket
def upload_pdf_to_gcp(pdf_path, blob_name):
    try:
        # Initialize GCP client and bucket
        bucket = initialize_gcp_client()

        # Create a blob object from the file and upload
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(pdf_path)
        print("*******************************************************************")
        # Instead of fetching public URL, construct the URL directly
        public_url = f"https://storage.googleapis.com/{bucket.name}/{blob_name}"

        # Return the constructed public URL
        return public_url

    except Exception as e:
        print(f"Error uploading PDF to GCP: {e}")
        return None


def process_pdf(pdf_path, segment_types, model_dir, output_dir):
    valid_segment_types = ["silt_fence", "rock_berms", "tree_protection", 
                           "stablized_construction", "inlet_protection"]

    plt.ion()
    warnings.filterwarnings('ignore')

    # Convert PDF to images
    image_paths = pdf_to_jpeg(pdf_path, output_dir)
    output_folder = "/home/dev/my_projects/MaskRCNN/verix-pro-v2/PDFs/result_test"  # Specify the output folder

    # Prepare output image paths
    output_image_paths = []

    for segment_name in segment_types:
        # Validate segment_name
        if segment_name not in valid_segment_types:
            print(f"########################Skipping invalid segment type: {segment_name}")
            continue  # Skip to the next segment type if invalid
        
        # Load the corresponding model


        model = load_model_for_segment(segment_name, model_dir)
        if model is None:
            print(f"Model not found for {segment_name}, skipping.")
            continue  # Skip this segment if model not found
        
        with open("data/JSON_files/types.json", 'r') as f:
            data = json.load(f)
        fence_types = data.get("types", [])
        print("!!!!!!fence typesssssssssssssssssssss : ", fence_types)
        class_names = ["BG"] + fence_types
        print("!!!!!!!!classssssssssss names : ", class_names)


        print("image paths : ",image_paths)
        # Loop through all files in the folder
        for file_name in os.listdir(output_dir):
            # Construct full file path
            file_path = os.path.join(output_dir, file_name)
            print("image name or path is : ",file_path)
            # Check if it's an image file
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Processing: {file_name}")

                # Read and preprocess the image
                image = cv2.imread(file_path)
                image_resized = resize_image(image)

                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization
                print("before inference image rgb : ",image_rgb)
                # Perform inference


                
                r = model.detect([image_rgb], verbose=1)[0]
                # Scale masks and bounding boxes back to match resized image dimensions


                original_shape = image.shape[:2]
                resized_masks = []
                original_mask_shape = r['masks'].shape[:2]  # (height, width)
                target_shape = image_resized.shape[:2]  # (new_height, new_width)

                for i in range(r['masks'].shape[-1]):
                    resized_mask = cv2.resize(r['masks'][:, :, i].astype(np.uint8),
                                            (target_shape[1], target_shape[0]),  # Ensure correct width, height order
                                            interpolation=cv2.INTER_NEAREST)  # Use nearest neighbor for segmentation masks
                    resized_masks.append(resized_mask)

                r['masks'] = np.stack(resized_masks, axis=-1) if resized_masks else np.zeros((*target_shape, 0), dtype=np.uint8)




                height_scale = image_resized.shape[0] / original_shape[0]
                width_scale = image_resized.shape[1] / original_shape[1]

                r['rois'] = np.round(r['rois'] * np.array([height_scale, width_scale, height_scale, width_scale])).astype(int)
                # Display results using visualize.display_instances (for comparison)
                plt.figure(figsize=(16, 16))  # Create a new figure
                display_instances(
                    image=image_rgb,
                    boxes=r['rois'], 
                    masks=r['masks'],
                    class_ids=r['class_ids'],
                    class_names=class_names,
                    scores=r['scores']
                )
                

                # Save the displayed figure as an image
                output_path_display = os.path.join(output_folder, f"display_{file_name}")
                #plt.savefig(output_path_display)  # Save the figure with boxes and masks

                # Close the plot to free memory
                plt.close()

                # Apply masks and draw bounding boxes directly on the original image
                for i in range(r['rois'].shape[0]):
                    mask = r['masks'][:, :, i]
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

                    # Apply mask to the original BGR image
                    for c in range(3):  # Apply mask channel-wise
                        image[:, :, c] = np.where(mask == 1,
                                                image[:, :, c] * 0.5 + color[c] * 0.5,
                                                image[:, :, c])

                    # Draw bounding box on the original BGR image
                    y1, x1, y2, x2 = r['rois'][i]
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                    # Add class label and score
                    class_id = r['class_ids'][i]
                    label = class_names[class_id] if class_id < len(class_names) else "Unknown"
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Save the OpenCV-processed image with masks and boxes
                output_path_cv = os.path.join(output_folder, f"processed_{file_name}")
                cv2.imwrite(output_path_cv, image)

    # Generate a final PDF from all annotated images
    #final_pdf_path = os.path.join(output_dir, "final_output.pdf")
    #create_pdf_from_images(output_image_paths, final_pdf_path)

    return True

    

# def process_pdf(pdf_path, segment_types, model_dir, output_dir):
#     valid_segment_types = ["silt_fence", "rock_berms", "tree_protection", 
#                            "stablized_construction", "inlet_protection"]

#     plt.ion()
#     warnings.filterwarnings('ignore')

#     # Convert PDF to images
#     image_paths = pdf_to_jpeg(pdf_path, output_dir)
#     output_folder = "/home/dev/my_projects/MaskRCNN/verix-pro-v2/PDFs/result_test"  # Specify the output folder

#     # Prepare output image paths
#     output_image_paths = []

#     for segment_name in segment_types:
#         # Validate segment_name
#         if segment_name not in valid_segment_types:
#             print(f"########################Skipping invalid segment type: {segment_name}")
#             continue  # Skip to the next segment type if invalid
        
#         # Load the corresponding model
#         model = load_model_for_segment(segment_name, model_dir)
#         if model is None:
#             print(f"Model not found for {segment_name}, skipping.")
#             continue  # Skip this segment if model not found
        
#         # Process each image for this segment type
#         for image_path in image_paths:
#             image = cv2.imread(image_path)

#             #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

#             # Run detection for the current segment model
#             print(f"##### Detecting {segment_name} in image {image_path}")
#             print("\n\n###################Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#             #r = model.detect([image], verbose=1)[0]

#             # Read and preprocess the image

#             # Perform inference

#             # Display results using visualize.display_instances (for comparison)
#             plt.figure(figsize=(16, 16))  # Create a new figure

#             # Load the fence types from JSON
#             with open("data/JSON_files/types.json", 'r') as f:
#                 data = json.load(f)
#             fence_types = data.get("types", [])
#             print("!!!!!!fence types : ", fence_types)
#             class_names = ["BG"] + fence_types
#             r = model.detect([image_rgb], verbose=1)[0]
#             print("!!!!!!!!clas names : ", class_names)


            

#             # Visualize the detected instances
#             display_instances(
#                 image=image_rgb,
#                 boxes=r['rois'],
#                 masks=r['masks'],
#                 class_ids=r['class_ids'],
#                 class_names=class_names,
#                 scores=r['scores']
#             )
#             plt.close()
#             # Save the result to the output directory
#             # output_image_path = os.path.join(output_dir, f"{segment_name}_inference_{os.path.basename(image_path)}")
#             # plt.savefig(output_image_path)
#             # output_image_paths.append(output_image_path)
#             output_path_display = os.path.join(output_folder, f"display_{image_path}")

#             # Apply masks and draw bounding boxes directly on the original image
#             for i in range(r['rois'].shape[0]):
#                 mask = r['masks'][:, :, i]
#                 color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

#                 # Apply mask to the original BGR image
#                 for c in range(3):  # Apply mask channel-wise
#                     image[:, :, c] = np.where(mask == 1,
#                                             image[:, :, c] * 0.5 + color[c] * 0.5,
#                                             image[:, :, c])

#                 # Draw bounding box on the original BGR image
#                 y1, x1, y2, x2 = r['rois'][i]
#                 cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

#                 # Add class label and score
#                 class_id = r['class_ids'][i]
#                 label = class_names[class_id] if class_id < len(class_names) else "Unknown"
#                 cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             # Save the OpenCV-processed image with masks and boxes
#             output_path_cv = os.path.join(output_folder, f"processed_{image_path}")
#             cv2.imwrite(output_path_cv, image)
#     # Generate a final PDF from all annotated images
#     final_pdf_path = os.path.join(output_dir, "final_output.pdf")
#     create_pdf_from_images(output_image_paths, final_pdf_path)

#     return final_pdf_path





# Function to process the downloaded PDF and return its public URL after uploading
def process_and_upload_pdf(pdf_url, items_to_detect, model_dir, output_dir):  #
    # Clean the URL
    clean_pdf_url = unquote(pdf_url).strip()
    # Extract the filename from the URL
    pdf_name = os.path.basename(urlparse(clean_pdf_url).path)
    
    # Now you have the filename in pdf_name
    print(f"############Input PDF filename: {pdf_name}")
    # Download the PDF
    response = requests.get(clean_pdf_url)

    if response.status_code == 200:
        print(f"Downloaded PDF from {clean_pdf_url}")
        if len(response.content) == 0:
            print(f"Error: Downloaded file is empty.")
            return None
        
        # Save the content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_pdf_path = tmp_file.name
            print(f"PDF saved to temporary file: {tmp_pdf_path}")

        # Process the PDF (you can implement your logic here)
        try:
            # Assuming process_pdf is a function that processes the PDF
            # and generates the output PDF
            final_pdf = process_pdf(tmp_pdf_path, items_to_detect, model_dir, output_dir)
            print(f"###########################################Processed PDF: {final_pdf}")

            # Upload the final PDF to GCP bucket
            file_name = os.path.basename(final_pdf)
            blob_name = f"processed_pdfs/{pdf_name}"

            public_url = upload_pdf_to_gcp(final_pdf, blob_name)

            # Return the public URL
            if public_url:
                return public_url
            else:
                print(f"Failed to upload PDF to GCP.")
                return None

        except Exception as e:
            print(f"Error processing PDF: {e}")
            return None

    else:
        print(f"Failed to download PDF from {clean_pdf_url} - Status Code: {response.status_code}")
        return None
