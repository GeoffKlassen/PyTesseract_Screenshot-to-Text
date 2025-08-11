import os
from PIL import Image
import numpy as np
import pandas as pd
import re

# Set working directory - At U of S
os.chdir('C:\\Users\\geoff\\OneDrive - University of Saskatchewan\\Grad Studies\\HappyB 2.0 2024\\R Scripts\\Input Files\\2025 Jan-')


# Function to calculate the mean squared error between two images
def calculate_mse(img1, img2):
    return np.mean((np.array(img1).flatten() - np.array(img2).flatten())**2)


# Directory containing the images
directory = 'Saved Images'

# List to store the results
results = []

# Get a list of all jpg files in the directory
# image_files = [f for f in os.listdir(directory)]


image_files = []
for root, _, files in os.walk(directory):
    for file in files:
        image_files.append(os.path.join(root, file))
prev_matches = np.full(len(image_files), 0)
image_groups = []

# Define a common size for all images
common_size = (120, 200)

# Attempt #1: handshake algorithm
"""
# Compare each image with every other image
for i, file1 in enumerate(image_files):
    count = prev_matches[i]  # if len(results) == 0 else results[i - 1][1] # this doesn't do what I thought it might
    image1 = Image.open(os.path.join(directory, file1)).resize(common_size)
    for j in range(i + 1, len(image_files)):
        file2 = image_files[j]
        image2 = Image.open(os.path.join(directory, file2)).resize(common_size)
        mse = calculate_mse(image1, image2)
        # Consider images as a close match if MSE is below a certain threshold (e.g., 100)
        if mse < 10:
            # print(f"{file1} {file2} {mse}")
            prev_matches[j] += 1
            count += 1
    results.append((file1, count))
    print(f"File #{i}: {file1}   Found {count} matches")
"""


# Attempt #2: comparing each file to the first file in each existing group

for i, file1 in enumerate(image_files):
    # if i > 400:
    #     break
    image1 = Image.open(os.path.join(file1)).resize(common_size)
    unique = True
    if len(image_groups) == 0:
        print(f"                                   Creating group 0 for file {i}")
        image_groups.append([file1])
    else:
        for j in range(len(image_groups) - 1, 0, -1):
            file2 = image_groups[j][0]
            image2 = Image.open(os.path.join(file2)).resize(common_size)
            mse = calculate_mse(image1, image2)
            if mse < 1:
                unique = False
                print(f"Adding file {i} to group {j}")
                image_groups[j].append(file1)
                break
        if unique:
            print(f"                                   Creating group {len(image_groups)} for file {i}")
            image_groups.append([file1])

pattern = re.compile(r'^[^-]+-[^-]+-([^-]+)-')
for i in range(len(image_files)):
    filename = image_files[i]
    avicenna_id = pattern.search(filename).group(1)
    image_url = "https://file.avicennaresearch.com/media/resp_files/" + str(avicenna_id) + "/image/" + filename
    for j in range(len(image_groups)):
        if filename in image_groups[j]:
            results.append((avicenna_id, image_url, j, len(image_groups[j])))
            break


# Create a DataFrame from the results
df = pd.DataFrame(results, columns=['Avicenna ID', 'File Name', 'Image Group Number', 'Appearance Count'])


df.to_csv('Duplicate Screenshots (working from laptop).csv')
