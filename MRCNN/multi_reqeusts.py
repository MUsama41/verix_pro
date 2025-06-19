import requests

# Define the base URL and payload structure
url = "http://18.222.249.206:5000/process_pdf"
headers = {"Content-Type": "application/json"}

# List of PDF URLs to use in the requests
pdf_urls = [
    #"https://storage.googleapis.com/weights_maskrcnn/silt_fence/silt_fence_pdf_c1.pdf",
    "https://storage.googleapis.com/weights_maskrcnn/silt_fence/silt_fence_pdf_c2.pdf",
    "https://storage.googleapis.com/weights_maskrcnn/silt_fence/silt_fence_pdf_c3.pdf",
    "https://storage.googleapis.com/weights_maskrcnn/silt_fence/silt_fence_pdf_c4.pdf",
    "https://storage.googleapis.com/weights_maskrcnn/silt_fence/silt_fence_pdf_c5.pdf",
    "https://storage.googleapis.com/weights_maskrcnn/silt_fence/silt_fence_pdf_c6.pdf",

]

# Send the requests
for pdf_url in pdf_urls:
    data = {
        "items_to_detect": ["silt_fence"],
        "pdf_urls": [
            {"id": 1, "url": pdf_url}
        ]
    }
    
    response = requests.get(url, json=data, headers=headers)

    # Print the response from the server
    if response.status_code == 200:
        print(f"Request with URL {pdf_url} succeeded!")
        print(response.json())  # Assuming the response is in JSON format
    else:
        print(f"Request with URL {pdf_url} failed. Status Code: {response.status_code}")
