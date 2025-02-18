import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Path to the NIfTI file
nifti_path = '/home/psaha03/scratch/training_data_k23/case_00003/imaging.nii.gz'

# Load the NIfTI image
nii = nib.load(nifti_path)
data = nii.get_fdata()

# Determine the number of slices in the axial direction (assumed axis=2)
num_slices = data.shape[2]

# Select 9 slices evenly spaced through the volume
slice_indices = np.linspace(0, num_slices - 1, 9, dtype=int)

# Create subplots to display the slices
fig, axes = plt.subplots(3, 3, figsize=(16, 16))
for i, ax in enumerate(axes.flat):
    slice_idx = slice_indices[i]
    ax.imshow(data[:, :, slice_idx], cmap='gray', origin='lower')
    ax.set_title(f"Slice {slice_idx}")
    ax.axis('off')

plt.tight_layout()
# Use plt.show() if you have an X-forwarding session, or save the figure:
plt.savefig('visualization_sagittal.png')
print("Visualization saved as 'visualization_sagittal.png'")
