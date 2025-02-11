import os
import json
import math

# Define the base directory where the images and labels are located
BASE_DIR = "/home/psaha03/scratch/dataset_brats24/dataset/training"

# Output JSON file names
OUTPUT_FILES = ["site-1.json", "site-2.json", "site-3.json", "site-4.json"]

# Initialize the dictionary to store the JSON structure for each site
site_data = [{"testing": [], "training": [], "validation": []} for _ in range(4)]

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

        # Check if label file exists
        if os.path.isfile(label_file):
            subjects.append({"image": image_files, "label": label_file})

# Shuffle the subjects for randomness
import random
random.shuffle(subjects)

# Divide subjects into four equal parts
chunk_size = math.ceil(len(subjects) / 4)
subject_chunks = [subjects[i:i + chunk_size] for i in range(0, len(subjects), chunk_size)]

# Process each chunk and divide into training and testing
for i, chunk in enumerate(subject_chunks):
    # Calculate split indices
    training_size = math.ceil(len(chunk) * 0.65)
    testing_size = len(chunk) - training_size

    # Split into training and testing
    site_data[i]["training"] = chunk[:training_size]
    site_data[i]["validation"] = chunk[training_size:]

# Write each site's data to separate JSON files
for i, site_json in enumerate(site_data):
    with open(OUTPUT_FILES[i], 'w') as outfile:
        json.dump(site_json, outfile, indent=4)
        
    print(f"Formatted JSON file generated: {OUTPUT_FILES[i]}")

