import os
import nibabel as nib
import json

dataset_path = "/home/psaha03/scratch/dataset_kits23/dataset/training/"
all_cases = sorted(os.listdir(dataset_path))

# Prepare a dictionary to store results
output = {}

# Initialize variables to track min and max shapes
min_shape = None
max_shape = None
min_case = ""
max_case = ""

for case in all_cases:
    image_path = os.path.join(dataset_path, case, "imaging.nii.gz")
    seg_path = os.path.join(dataset_path, case, "segmentation.nii.gz")
    
    if os.path.exists(image_path) and os.path.exists(seg_path):
        image_shape = nib.load(image_path).shape
        seg_shape = nib.load(seg_path).shape
        
        if image_shape == seg_shape:
            output[case] = {"status": "Shapes match", "image_shape": image_shape}
        else:
            output[case] = {"status": "Shape mismatch", "image_shape": image_shape, "seg_shape": seg_shape}
        
        # Track min and max shapes
        if min_shape is None or image_shape < min_shape:
            min_shape = image_shape
            min_case = case
        if max_shape is None or image_shape > max_shape:
            max_shape = image_shape
            max_case = case
    else:
        output[case] = {"status": "Missing file", "image_file": os.path.exists(image_path), "seg_file": os.path.exists(seg_path)}

# Include the min and max shape information in the output
output["min_shape"] = {"case": min_case, "shape": min_shape}
output["max_shape"] = {"case": max_case, "shape": max_shape}

# Save the output to a JSON file
output_file = "/home/psaha03/scratch/safeseg/src/kits23_shapes.json"
with open(output_file, 'w') as json_file:
    json.dump(output, json_file, indent=4)

print(f"Results saved to {output_file}")
