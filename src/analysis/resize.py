import os
import nibabel as nib
import numpy as np
import scipy.ndimage

# Define the target shape (adjust based on your model's input size)
TARGET_SHAPE = (168, 192, 192)  

def resize_image(image, target_shape, interpolation_order):
    """Resize a NIfTI image to the target shape."""
    current_shape = image.shape
    scale_factors = np.array(target_shape) / np.array(current_shape)

    # Resize image data
    resized_data = scipy.ndimage.zoom(image.get_fdata(), scale_factors, order=interpolation_order)

    # Update affine to reflect new voxel size
    new_affine = image.affine.copy()
    for i in range(3):
        new_affine[i, i] *= (current_shape[i] / target_shape[i])

    return nib.Nifti1Image(resized_data, new_affine, image.header)

# Input and output directories
input_folder = "/home/psaha03/scratch/kits_small"
output_folder = "/home/psaha03/scratch/kits_small_resized"

os.makedirs(output_folder, exist_ok=True)

for case in sorted(os.listdir(input_folder)):
    case_path = os.path.join(input_folder, case)
    image_path = os.path.join(case_path, "imaging.nii.gz")
    seg_path = os.path.join(case_path, "segmentation.nii.gz")
    
    if os.path.exists(image_path) and os.path.exists(seg_path):
        print(f"Processing {case}...")

        # Load images
        img_nifti = nib.load(image_path)
        seg_nifti = nib.load(seg_path)

        # Resize image (trilinear interpolation)
        resized_img = resize_image(img_nifti, TARGET_SHAPE, interpolation_order=1)

        # Resize segmentation (nearest-neighbor interpolation)
        resized_seg = resize_image(seg_nifti, TARGET_SHAPE, interpolation_order=0)

        # Save resized images
        case_output_folder = os.path.join(output_folder, case)
        os.makedirs(case_output_folder, exist_ok=True)

        nib.save(resized_img, os.path.join(case_output_folder, "imaging.nii.gz"))
        nib.save(resized_seg, os.path.join(case_output_folder, "segmentation.nii.gz"))

        print(f"Saved resized images for {case} in {case_output_folder}")

print("Resizing complete.")
