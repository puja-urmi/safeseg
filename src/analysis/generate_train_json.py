import os
import json
import random

# Define dataset path
dataset_path = "/home/psaha03/scratch/dataset_kits23/dataset/training/"

# Get list of all cases
all_cases = sorted(os.listdir(dataset_path))  # Ensure sorted for reproducibility
random.shuffle(all_cases)  # Shuffle to randomize split

# Split dataset into 4 non-overlapping parts
num_sites = 3
cases_per_site = len(all_cases) // num_sites
sites = [all_cases[i * cases_per_site: (i + 1) * cases_per_site] for i in range(num_sites)]

# Generate JSON files for each site
for site_idx, site_cases in enumerate(sites, start=1):
    random.shuffle(site_cases)  # Shuffle cases within the site
    split_idx = int(len(site_cases) * 0.75)  # 75% for training
    
    training_cases = site_cases[:split_idx]
    validation_cases = site_cases[split_idx:]
    
    dataset_json = {"testing": [], "training": [], "validation": []}
    
    for case in training_cases:
        dataset_json["training"].append({
            "image": [f"training/{case}/imaging.nii.gz"],
            "label": f"training/{case}/segmentation.nii.gz"
        })
    
    for case in validation_cases:
        dataset_json["validation"].append({
            "image": [f"training/{case}/imaging.nii.gz"],
            "label": f"training/{case}/segmentation.nii.gz"
        })
    
    # Save JSON file
    json_filename = f"site-{site_idx}.json"
    with open(json_filename, "w") as json_file:
        json.dump(dataset_json, json_file, indent=4)
    
    print(f"Generated {json_filename} with {len(training_cases)} training and {len(validation_cases)} validation cases.")
