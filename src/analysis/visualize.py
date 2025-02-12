import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_segmentation(image_path, seg_path, slice_index=None):
    # Load NIfTI images
    image_nii = nib.load(image_path)
    seg_nii = nib.load(seg_path)
    
    # Convert to numpy arrays
    image_data = image_nii.get_fdata()
    seg_data = seg_nii.get_fdata()
    
    # Determine slice index if not provided (use the middle slice along the axial plane)
    if slice_index is None:
        slice_index = image_data.shape[2] // 2
    
    # Extract the selected slice
    image_slice = image_data[:, :, slice_index]
    seg_slice = seg_data[:, :, slice_index]
    
    # Plot the image and segmentation overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(image_slice, cmap='gray')
    plt.imshow(seg_slice, cmap='jet', alpha=0.5)  # Overlay segmentation with transparency
    plt.colorbar(label='Segmentation Label')
    plt.title(f'Segmentation Overlay - Slice {slice_index}')
    plt.axis('off')
    plt.show()

# Paths to your data
image_path = "/home/psaha03/scratch/dataset_kits23/dataset/training/case_00005/imaging.nii.gz"
seg_path = "/home/psaha03/scratch/dataset_kits23/dataset/training/case_00005/segmentation.nii.gz"

# Run visualization
visualize_segmentation(image_path, seg_path)