import h5py

file_path = "data/weights/rock_berms.h5"

try:
    with h5py.File(file_path, "r") as f:
        print(f"{file_path} loaded successfully.")
except Exception as e:
    print(f"Error loading {file_path}: {e}")
