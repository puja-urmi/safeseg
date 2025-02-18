import os
import nibabel as nib
import numpy as np
import shutil

def scale_image(image_data):
    """
    Scales the image intensities to [-1, 1] based on the per-case min and max.
    
    Args:
        image_data (numpy.ndarray): The input image array.
        
    Returns:
        numpy.ndarray: Processed image data after scaling.
    """
    # Compute per-case minimum and maximum
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    
    if max_val == min_val:
        raise ValueError("Image intensity range is zero; cannot scale.")
    
    # Scale intensities to the [-1, 1] range
    scaled = 2 * (image_data - min_val) / (max_val - min_val) - 1
    return scaled

# Define the source and destination directories
src_root = '/home/psaha03/scratch/resized'
dst_root = '/home/psaha03/scratch/rescaled'
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

    # Process scaling
    try:
        processed_data = scale_image(image_data)
    except ValueError as e:
        print(f"Skipping {case}: {e}")
        continue

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
