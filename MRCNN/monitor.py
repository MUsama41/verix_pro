import os
from google.cloud import storage
from google.oauth2 import service_account
import time
import re

# Path to your service account JSON file
key_path = '/home/ubuntu/maskrcnn/verix_pro_ai/data/JSON_files/GCP_json.json'  # Replace with your credentials file path

# Authenticate using the key
credentials = service_account.Credentials.from_service_account_file(key_path)

# Initialize the Google Cloud Storage client
client = storage.Client(credentials=credentials, project='level-sol-440022-t2')  # Replace with your project ID

# Access the bucket
bucket = client.get_bucket('inlet_protection')  # Replace with your bucket name

# Directory to monitor
#LOGS_DIR = '/home/ubuntu/verispro/verix_pro_ai/logs'  # Replace with the actual logs directory path
LOGS_DIR = '/home/ubuntu/maskrcnn/verix_pro_ai/logs'  # Updated with the new path



# Set to store already uploaded files (avoiding duplicates)
uploaded_files = set()

# Function to upload files to GCP
def upload_to_gcp(local_file_path):
    destination_blob_name = os.path.basename(local_file_path)  # Extract file name
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"File uploaded to {bucket.name}/{destination_blob_name}.")
    # Add the file to the uploaded_files set to prevent future uploads
    uploaded_files.add(local_file_path)

# Function to recursively check subfolders for .h5 files
def check_and_upload_subfolders(directory):
    for root, dirs, files in os.walk(directory):
        # For each file in the directory
        for filename in files:
            if re.match(r'mask_rcnn_object_\d+\.h5$', filename):
                local_file_path = os.path.join(root, filename)
                # Only upload the file if it hasn't been uploaded already
                if local_file_path not in uploaded_files:
                    upload_to_gcp(local_file_path)

# Monitor the logs directory for new .h5 files in subfolders continuously
print("Monitoring logs directory for new .h5 files...")
while True:
    check_and_upload_subfolders(LOGS_DIR)
    
    # Sleep for a few seconds before checking again
    time.sleep(5)  # Adjust sleep time as necessary
