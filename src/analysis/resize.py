import os
import nibabel as nib
import numpy as np
from monai.transforms import Resize

# Define the target shape
TARGET_SHAPE = (200, 250, 250)

# Define MONAI resize transforms
resize_image_transform = Resize(spatial_size=TARGET_SHAPE, mode='bilinear')
resize_seg_transform = Resize(spatial_size=TARGET_SHAPE, mode='nearest')

def resize_image_monai(image, transform):
    """Resize a NIfTI image using MONAI."""
    resized_data = transform(image.get_fdata()[None, None, ...])  # Add batch & channel dims
    return nib.Nifti1Image(resized_data[0, 0].numpy(), image.affine, image.header)

# Input and output directories
input_folder = "/home/psaha03/scratch/dataset_kits23/dataset/training"
output_folder = "/home/psaha03/scratch/training_k23_c+resized"

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
        resized_img = resize_image_monai(img_nifti, resize_image_transform)

        # Resize segmentation (nearest-neighbor interpolation)
        resized_seg = resize_image_monai(seg_nifti, resize_seg_transform)

        # Save resized images
        case_output_folder = os.path.join(output_folder, case)
        os.makedirs(case_output_folder, exist_ok=True)

        nib.save(resized_img, os.path.join(case_output_folder, "imaging.nii.gz"))
        nib.save(resized_seg, os.path.join(case_output_folder, "segmentation.nii.gz"))

        print(f"Saved resized images for {case} in {case_output_folder}")

print("Resizing complete.")
