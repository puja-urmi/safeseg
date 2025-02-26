import os
import json

# Define dataset path
dataset_path = "/home/psaha03/scratch/dataset_kits23/dataset/validation"

# Get list of all cases
all_cases = sorted(os.listdir(dataset_path))  

# Create JSON structure
dataset_json = {"testing": [], "training": [], "validation": []}

# Add all cases to validation
for case in all_cases:
    dataset_json["validation"].append({
        "image": [f"training/{case}/imaging.nii.gz"],
        "label": f"training/{case}/segmentation.nii.gz"
    })

# Save JSON file
json_filename = "site-test.json"
with open(json_filename, "w") as json_file:
    json.dump(dataset_json, json_file, indent=4)

print(f"Generated {json_filename} with {len(all_cases)} validation cases.")
