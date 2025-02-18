import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Path to the NIfTI file
nifti_path = '/home/psaha03/scratch/equalized+sigmoided/case_00003/imaging.nii.gz'

# Load the NIfTI image
nii = nib.load(nifti_path)
data = nii.get_fdata()

# For sagittal slices, use the first dimension (X-axis)
num_slices = data.shape[0]

# Select 9 slices evenly spaced through the volume along the sagittal plane
slice_indices = np.linspace(0, num_slices - 1, 9, dtype=int)

# Create subplots to display the slices
fig, axes = plt.subplots(3, 3, figsize=(16, 16))
for i, ax in enumerate(axes.flat):
    slice_idx = slice_indices[i]
    # Extract the sagittal slice (X-axis) and adjust orientation if needed
    sagittal_slice = data[slice_idx, :, :]
    ax.imshow(np.rot90(sagittal_slice), cmap='gray', origin='lower')
    ax.set_title(f"Sagittal Slice {slice_idx}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('visualization_axial_equalized+sigmoided.png')
print("Visualization saved as 'visualization_axial.png'")
