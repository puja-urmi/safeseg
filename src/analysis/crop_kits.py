import os
import nibabel as nib
import numpy as np

# Define source and destination root directories
src_root = '/home/psaha03/scratch/training_k23_sampled'
dst_root = '/home/psaha03/scratch/training_k23_sampled+cropped'

def crop_to_label(image_data, seg_data):
    # Find indices where segmentation is non-zero
    coords = np.argwhere(seg_data)
    if coords.size == 0:
        raise ValueError("Segmentation mask is empty!")
    # Determine bounding box
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 2  # add 10 for flexible use
    # Crop both image and seg using the bounding box
    slicer = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
    return image_data[slicer], seg_data[slicer]

# Iterate over case folders
for case in os.listdir(src_root):
    case_path = os.path.join(src_root, case)
    if not os.path.isdir(case_path):
        continue  # Skip if not a directory

    # Define file paths for image and segmentation
    image_path = os.path.join(case_path, 'imaging.nii.gz')
    seg_path   = os.path.join(case_path, 'segmentation.nii.gz')

    if not (os.path.exists(image_path) and os.path.exists(seg_path)):
        print(f"Skipping {case} as required files are missing.")
        continue

    # Load image and segmentation
    image_nii = nib.load(image_path)
    seg_nii = nib.load(seg_path)
    image_data = image_nii.get_fdata()
    seg_data = seg_nii.get_fdata()

    # Crop to region with labels
    try:
        cropped_image, cropped_seg = crop_to_label(image_data, seg_data)
    except ValueError as e:
        print(f"Skipping {case}: {e}")
        continue

    # Create new Nifti images preserving original affine and header if needed
    cropped_image_nii = nib.Nifti1Image(cropped_image, image_nii.affine, image_nii.header)
    cropped_seg_nii = nib.Nifti1Image(cropped_seg, seg_nii.affine, seg_nii.header)

    # Create destination directory for this case if it doesn't exist
    dst_case_path = os.path.join(dst_root, case)
    os.makedirs(dst_case_path, exist_ok=True)

    # Save cropped images
    nib.save(cropped_image_nii, os.path.join(dst_case_path, 'imaging.nii.gz'))
    nib.save(cropped_seg_nii, os.path.join(dst_case_path, 'segmentation.nii.gz'))
    print(f"Processed and saved {case}")
