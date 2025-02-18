import os
import nibabel as nib
import numpy as np
import shutil
import cv2

def apply_histogram_equalization(image_data):
    """
    Applies histogram equalization slice-wise to a 3D medical image.
    
    Args:
        image_data (numpy.ndarray): The input 3D image array.
    
    Returns:
        numpy.ndarray: The histogram-equalized image.
    """
    equalized_image = np.zeros_like(image_data)
    
    for i in range(image_data.shape[0]):  # Apply HE slice-wise along the axial plane
        slice_ = image_data[i, :, :]
        slice_ = cv2.normalize(slice_, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Normalize to [0,255]
        equalized_image[i, :, :] = cv2.equalizeHist(slice_)
    
    return equalized_image

# Define source and destination directories
src_root = '/home/psaha03/scratch/resized'
dst_root = '/home/psaha03/scratch/equalized'
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

    # Load imaging file
    image_nii = nib.load(image_path)
    image_data = image_nii.get_fdata()

    # Apply histogram equalization
    processed_data = apply_histogram_equalization(image_data)

    # Create a new Nifti image preserving the original affine and header
    new_nii = nib.Nifti1Image(processed_data, image_nii.affine, image_nii.header)

    # Create destination directory for the case and save the processed image
    dst_case_dir = os.path.join(dst_root, case)
    os.makedirs(dst_case_dir, exist_ok=True)
    imaging_output_path = os.path.join(dst_case_dir, 'imaging.nii.gz')
    nib.save(new_nii, imaging_output_path)
    print(f"Processed and saved imaging for {case} at {imaging_output_path}")

    # Copy segmentation file without any processing
    seg_path = os.path.join(case_path, 'segmentation.nii.gz')
    if os.path.exists(seg_path):
        dst_seg_path = os.path.join(dst_case_dir, 'segmentation.nii.gz')
        shutil.copy(seg_path, dst_seg_path)
        print(f"Copied segmentation for {case} to {dst_seg_path}")
    else:
        print(f"Segmentation file missing for {case}.")