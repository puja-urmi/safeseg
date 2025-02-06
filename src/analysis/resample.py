import nibabel as nib
import numpy as np
import scipy.ndimage
import os

def resample_image(image, target_voxel_size, interpolation_order):
    """Resample a NIfTI image to the target voxel size."""
    affine = image.affine

    try:
        current_voxel_size = nib.affines.voxel_sizes(affine)  # More robust voxel size extraction
    except Exception as e:
        raise ValueError(f"Error extracting voxel size from affine:\n{affine}\nError: {e}")

    resampling_factor = np.array(current_voxel_size) / target_voxel_size
    new_shape = np.round(image.shape * resampling_factor).astype(int)

    resampled_data = scipy.ndimage.zoom(image.get_fdata(), resampling_factor, order=interpolation_order)

    # Construct a new affine matrix
    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] = affine[i, i] * (target_voxel_size[i] / current_voxel_size[i])

    return nib.Nifti1Image(resampled_data, new_affine, image.header)


target_voxel_size = np.array([0.5, 0.5, 1.5])  # Adjust if needed
input_folder = "/home/psaha03/scratch/oversampled"
output_folder = "/home/psaha03/scratch/resampled"

os.makedirs(output_folder, exist_ok=True)

for case in sorted(os.listdir(input_folder)):
    case_path = os.path.join(input_folder, case)
    image_path = os.path.join(case_path, "imaging.nii.gz")
    seg_path = os.path.join(case_path, "segmentation.nii.gz")
    
    if os.path.exists(image_path) and os.path.exists(seg_path):
        print(f"Processing {case}...")
        
        # Load NIfTI images
        img_nifti = nib.load(image_path)
        seg_nifti = nib.load(seg_path)
        
        # Resample imaging with trilinear interpolation (order=1)
        resampled_img = resample_image(img_nifti, target_voxel_size, interpolation_order=1)
        
        # Resample segmentation with nearest-neighbor interpolation (order=0)
        resampled_seg = resample_image(seg_nifti, target_voxel_size, interpolation_order=0)
        
        # Save resampled images
        case_output_folder = os.path.join(output_folder, case)
        os.makedirs(case_output_folder, exist_ok=True)
        
        nib.save(resampled_img, os.path.join(case_output_folder, "imaging.nii.gz"))
        nib.save(resampled_seg, os.path.join(case_output_folder, "segmentation.nii.gz"))
        
        print(f"Saved resampled images for {case} in {case_output_folder}")

print("Resampling complete.")
