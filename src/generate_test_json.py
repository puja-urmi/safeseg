import os
import json
import random

# Define the base directory where the images and labels are located
BASE_DIR = "validation"

# Output JSON file name
OUTPUT_FILE = "site-test.json"

# Initialize the dictionary to store the JSON structure
site_data = {"training": [], "testing": [], "validation": []}

# Gather all subjects from the base directory
subjects = []
for subject_dir in os.listdir(BASE_DIR):
    subject_path = os.path.join(BASE_DIR, subject_dir)
    if os.path.isdir(subject_path):
        # Define the list of modalities
        modalities = ['t1c', 't1n', 't2f', 't2w']

        # Create the list of image files (for each modality)
        image_files = []
        for modality in modalities:
            image_file = os.path.join(BASE_DIR, subject_dir, f"{subject_dir}-{modality}.nii.gz")
            if os.path.isfile(image_file):
                image_files.append(image_file)

        # Find the label file (segmentation)
        label_file = os.path.join(BASE_DIR, subject_dir, f"{subject_dir}-seg.nii.gz")

        # Check if label file exists and all modalities exist
        if os.path.isfile(label_file) and len(image_files) == len(modalities):
            subject_entry = {"image": image_files, "label": label_file}
            subjects.append(subject_entry)
        else:
            print(f"Skipping {subject_dir} due to missing files")

# Shuffle the subjects for randomness
random.shuffle(subjects)

# Place all subjects in the validation set
site_data["validation"] = subjects

# Write the data to the JSON file
with open(OUTPUT_FILE, 'w') as outfile:
    json.dump(site_data, outfile, indent=4)

print(f"Formatted JSON file generated: {OUTPUT_FILE} with {len(subjects)} validation subjects.")
