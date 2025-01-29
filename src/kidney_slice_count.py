import os
import nibabel as nib
import numpy as np

# Define dataset path and case number
dataset_path = "/home/psaha03/scratch/dataset_kits23/dataset/training"
case_number = "case_00151"
image_path = os.path.join(dataset_path, case_number, "imaging.nii.gz")
seg_path = os.path.join(dataset_path, case_number, "segmentation.nii.gz")

# Load the image and segmentation data
image_data = nib.load(image_path).get_fdata()
seg_data = nib.load(seg_path).get_fdata()

# Find slices with kidney label (label = 1)
kidney_slices = []
for slice_idx in range(image_data.shape[2]):  # Iterate over the third dimension (slices)
    slice_seg = seg_data[:, :, slice_idx]
    if np.any(slice_seg == 1):  # Check if kidney label (1) is present in the slice
        kidney_slices.append(slice_idx)

# Find the range of slices containing the kidney label
if kidney_slices:
    first_slice = kidney_slices[0]
    last_slice = kidney_slices[-1]
    print(f"Kidney is present between slices {first_slice} and {last_slice}.")
    print(f"Total number of slices containing kidney: {len(kidney_slices)}")
else:
    print("No kidney label found in this case.")
