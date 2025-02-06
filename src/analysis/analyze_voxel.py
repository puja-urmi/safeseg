import os
import nibabel as nib
import numpy as np
import json

dataset_path = "/home/psaha03/scratch/training_data_k23"
all_cases = sorted(os.listdir(dataset_path))

# Define the target voxel size (example)
target_voxel_size = np.array([0.5, 0.5, 1.5])  # in mm: [height, width, depth]

# Initialize lists to store results
oversampled_cases = []
undersampled_cases = []

for case in all_cases:
    image_path = os.path.join(dataset_path, case, "imaging.nii.gz")
    
    if os.path.exists(image_path):
        # Load the image and get voxel size from affine matrix
        img = nib.load(image_path)
        affine = img.affine
        voxel_size = np.abs(np.diag(affine)[:3])  # Extract voxel size (height, width, depth)
        
        # Compare voxel size to the target voxel size
        if np.all(voxel_size < target_voxel_size):  # Oversampled
            oversampled_cases.append((case, voxel_size.tolist()))
        elif np.all(voxel_size > target_voxel_size):  # Undersampled
            undersampled_cases.append((case, voxel_size.tolist()))

# Print total counts
print(f"Total Oversampled Cases: {len(oversampled_cases)}")
print(f"Total Undersampled Cases: {len(undersampled_cases)}")

# Save results
oversampled_file = "/home/psaha03/scratch/oversampled_cases.json"
undersampled_file = "/home/psaha03/scratch/undersampled_cases.json"

with open(oversampled_file, 'w') as f:
    json.dump(oversampled_cases, f, indent=4)

with open(undersampled_file, 'w') as f:
    json.dump(undersampled_cases, f, indent=4)

print(f"Results saved to {oversampled_file} and {undersampled_file}")
