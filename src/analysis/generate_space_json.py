import os
import nibabel as nib
import json

# Path to the dataset
input_folder = "/home/psaha03/scratch/dataset_kits23/dataset/training/"

# Function to extract voxel spacing
def get_voxel_spacing(image_path):
    """Extract voxel spacing from the NIfTI image."""
    img_nifti = nib.load(image_path)
    spacing = img_nifti.header.get_zooms()  # Get voxel size (spacing)
    # Convert to standard Python float types for JSON serialization
    return tuple(float(val) for val in spacing)

# Dictionary to hold voxel spacing for all cases
all_voxel_spacing = {}

# Loop through each case in the dataset
for case in sorted(os.listdir(input_folder)):
    case_path = os.path.join(input_folder, case)
    image_path = os.path.join(case_path, "imaging.nii.gz")
    
    # Check if the image file exists
    if os.path.exists(image_path):
        print(f"Processing {case}...")
        
        # Extract voxel spacing
        spacing = get_voxel_spacing(image_path)
        
        # Add to dictionary
        all_voxel_spacing[case] = {
            "voxel_spacing": {
                "x": spacing[0],  # Voxel size in X direction
                "y": spacing[1],  # Voxel size in Y direction
                "z": spacing[2]   # Voxel size in Z direction
            }
        }

# Path for the single JSON file
output_json_path = "/home/psaha03/scratch/voxel_spacing_all_cases.json"

# Save the entire voxel spacing data to a single JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(all_voxel_spacing, json_file, indent=4)

print(f"Saved voxel spacing for all cases in {output_json_path}")
