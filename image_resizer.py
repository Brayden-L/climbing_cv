# %%

import os
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch

def resize_and_pad_images(input_folder, output_folder, target_size=(224, 224)):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define the transformation for resizing
    resize_transform = transforms.Compose([
        transforms.Resize(target_size),
    ])

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_folder, file)

            try:
                # Open the image file
                with Image.open(input_path) as img:
                    # Apply the resizing transformation
                    resized_img = resize_transform(img)

                    # Save the resized and padded image to the output folder
                    resized_img.save(output_path)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

input_folder = r"C:\Users\Brayden\Desktop\climbing_photo_rear_glory_topo_classifier\climbing_classifier_data\topo"  # Replace with the path to your input folder
output_folder = r"C:\Users\Brayden\Desktop\climbing_photo_rear_glory_topo_classifier\climbing_classifier_data\topo_mod"  # Replace with the path to your output folder
resize_and_pad_images(input_folder, output_folder)

# %%
