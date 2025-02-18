import os
import nibabel as nib
import numpy as np

# Define the root directory containing case folders
src_root = '/home/psaha03/scratch/training_data_k23'

# Initialize global min and max values
global_min = float('inf')
global_max = float('-inf')

# Iterate over each case folder
for case in os.listdir(src_root):
    case_path = os.path.join(src_root, case)
    if not os.path.isdir(case_path):
        continue

    # Define the path to the imaging file
    image_path = os.path.join(case_path, 'imaging.nii.gz')
    if not os.path.exists(image_path):
        print(f"Skipping {case} as imaging file is missing.")
        continue

    # Load the image and retrieve intensity data
    image_nii = nib.load(image_path)
    image_data = image_nii.get_fdata()

    # Compute the case-specific min and max
    case_min = np.min(image_data)
    case_max = np.max(image_data)
    print(f"Case: {case} - Min: {case_min}, Max: {case_max}")

    # Update the global min and max
    global_min = min(global_min, case_min)
    global_max = max(global_max, case_max)

print(f"\nGlobal minimum intensity: {global_min}")
print(f"Global maximum intensity: {global_max}")
