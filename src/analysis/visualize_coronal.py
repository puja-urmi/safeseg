import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Path to the NIfTI file
nifti_path = '/home/psaha03/scratch/training_data_k23/case_00003/imaging.nii.gz'

# Load the NIfTI image
nii = nib.load(nifti_path)
data = nii.get_fdata()

# For coronal slices, the slice index corresponds to the second dimension (Y-axis)
num_slices = data.shape[1]

# Select 9 slices evenly spaced through the volume along the coronal plane
slice_indices = np.linspace(0, num_slices - 1, 9, dtype=int)

# Create subplots to display the slices
fig, axes = plt.subplots(3, 3, figsize=(16, 16))
for i, ax in enumerate(axes.flat):
    slice_idx = slice_indices[i]
    # Extract the coronal slice
    coronal_slice = data[:, slice_idx, :]
    ax.imshow(coronal_slice.T, cmap='gray', origin='lower')  # Transpose if needed for correct orientation
    ax.set_title(f"Coronal Slice {slice_idx}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('visualization_coronal.png')
print("Visualization saved as 'visualization_coronal.png'")
