import os
import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage

# Define input and output directories
src_root = '/home/psaha03/scratch/resampled'
dst_root = '/home/psaha03/scratch/resized'
os.makedirs(dst_root, exist_ok=True)

# Define target shape (modify as needed)
target_shape = (150, 150, 150)

# Process each case folder in src_root
for case in os.listdir(src_root):
    case_path = os.path.join(src_root, case)
    if not os.path.isdir(case_path):
        continue

    # Define file paths
    imaging_path = os.path.join(case_path, 'imaging.nii.gz')
    seg_path = os.path.join(case_path, 'segmentation.nii.gz')

    if not os.path.exists(imaging_path) or not os.path.exists(seg_path):
        print(f"Skipping {case}: required files are missing.")
        continue

    # Load imaging and segmentation
    imaging_nii = nib.load(imaging_path)
    seg_nii = nib.load(seg_path)
    imaging_data = imaging_nii.get_fdata()
    seg_data = seg_nii.get_fdata()

    # Calculate zoom factors for each dimension
    zoom_factors = [target_shape[i] / imaging_data.shape[i] for i in range(3)]

    # Resize imaging with cubic interpolation (order=3)
    imaging_resized = ndimage.zoom(imaging_data, zoom_factors, order=3)
    # Resize segmentation with nearest-neighbor interpolation (order=0)
    seg_resized = ndimage.zoom(seg_data, zoom_factors, order=0)

    # Update the affine: new voxel size = original voxel size / zoom factor
    new_affine = imaging_nii.affine.copy()
    for i in range(3):
        new_affine[i, i] = imaging_nii.affine[i, i] / zoom_factors[i]

    # Create new NIfTI images
    imaging_resized_nii = nib.Nifti1Image(imaging_resized, new_affine, imaging_nii.header)
    seg_resized_nii = nib.Nifti1Image(seg_resized, new_affine, seg_nii.header)

    # Create destination folder for this case
    dst_case_path = os.path.join(dst_root, case)
    os.makedirs(dst_case_path, exist_ok=True)

    # Save the resized images
    imaging_out_path = os.path.join(dst_case_path, 'imaging.nii.gz')
    seg_out_path = os.path.join(dst_case_path, 'segmentation.nii.gz')
    nib.save(imaging_resized_nii, imaging_out_path)
    nib.save(seg_resized_nii, seg_out_path)

    print(f"Resized and saved {case}")
