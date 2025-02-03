import os
import nibabel as nib
import json

dataset_path = "/home/psaha03/scratch/training"
all_cases = sorted(os.listdir(dataset_path))

# Prepare a dictionary to store results
output = {}

# Initialize min and max trackers for each dimension
min_shape = [None, None, None]  # [min_height, min_width, min_depth]
max_shape = [None, None, None]  # [max_height, max_width, max_depth]
min_case = ["", "", ""]
max_case = ["", "", ""]

for case in all_cases:
    image_path = os.path.join(dataset_path, case, "imaging.nii.gz")
    seg_path = os.path.join(dataset_path, case, "segmentation.nii.gz")
    
    if os.path.exists(image_path) and os.path.exists(seg_path):
        image_shape = nib.load(image_path).shape  # (Height, Width, Depth)
        seg_shape = nib.load(seg_path).shape
        
        if image_shape == seg_shape:
            output[case] = {"status": "Shapes match", "image_shape": image_shape}
        else:
            output[case] = {"status": "Shape mismatch", "image_shape": image_shape, "seg_shape": seg_shape}
        
        # Track min and max separately for each dimension
        for i in range(3):  # Iterate over height, width, depth
            if min_shape[i] is None or image_shape[i] < min_shape[i]:
                min_shape[i] = image_shape[i]
                min_case[i] = case
            if max_shape[i] is None or image_shape[i] > max_shape[i]:
                max_shape[i] = image_shape[i]
                max_case[i] = case
    else:
        output[case] = {"status": "Missing file", "image_file": os.path.exists(image_path), "seg_file": os.path.exists(seg_path)}

# Include the min and max shape information in the output
output["min_shape"] = {
    "height": {"case": min_case[0], "value": min_shape[0]},
    "width": {"case": min_case[1], "value": min_shape[1]},
    "depth": {"case": min_case[2], "value": min_shape[2]}
}

output["max_shape"] = {
    "height": {"case": max_case[0], "value": max_shape[0]},
    "width": {"case": max_case[1], "value": max_shape[1]},
    "depth": {"case": max_case[2], "value": max_shape[2]}
}

# Save the output to a JSON file
output_file = "/home/psaha03/scratch/safeseg/src/analysis/kits23_new_shapes.json"
with open(output_file, 'w') as json_file:
    json.dump(output, json_file, indent=4)

print(f"Results saved to {output_file}")
