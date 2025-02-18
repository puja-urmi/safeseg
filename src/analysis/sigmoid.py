import os
import nibabel as nib
import numpy as np
import shutil

def apply_sigmoid(image_data):
    """
    Applies the sigmoid function to the image intensities.

    Args:
        image_data (numpy.ndarray): The input image array.

    Returns:
        numpy.ndarray: Processed image data after sigmoid transformation.
    """
    return 1 / (1 + np.exp(-image_data))

# Define the source and destination directories
src_root = '/home/psaha03/scratch/equalized'
dst_root = '/home/psaha03/scratch/equalized+sigmoided'
os.makedirs(dst_root, exist_ok=True)

# Process each case
for case in os.listdir(src_root):
    case_path = os.path.join(src_root, case)
    if not os.path.isdir(case_path):
        continue

    image_path = os.path.join(case_path, 'imaging.nii.gz')
    if not os.path.exists(image_path):
        print(f"Skipping {case} as imaging file is missing.")
        continue

    # Load the imaging file
    image_nii = nib.load(image_path)
    image_data = image_nii.get_fdata()

    # Apply the sigmoid transformation
    processed_data = apply_sigmoid(image_data)

    # Create a new Nifti image preserving the original affine and header
    new_nii = nib.Nifti1Image(processed_data, image_nii.affine, image_nii.header)

    # Create destination directory for the case and save the processed image
    dst_case_dir = os.path.join(dst_root, case)
    os.makedirs(dst_case_dir, exist_ok=True)
    imaging_output_path = os.path.join(dst_case_dir, 'imaging.nii.gz')
    nib.save(new_nii, imaging_output_path)
    print(f"Processed and saved imaging for {case} at {imaging_output_path}")

    # Copy the segmentation file without any processing
    seg_path = os.path.join(case_path, 'segmentation.nii.gz')
    if os.path.exists(seg_path):
        dst_seg_path = os.path.join(dst_case_dir, 'segmentation.nii.gz')
        shutil.copy(seg_path, dst_seg_path)
        print(f"Copied segmentation for {case} to {dst_seg_path}")
    else:
        print(f"Segmentation file missing for {case}.")
