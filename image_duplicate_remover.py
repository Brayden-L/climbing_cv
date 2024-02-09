# %%
from PIL import Image
import imagehash
import os
from collections import defaultdict

# %%
folder_path = r"C:\Users\Brayden\Desktop\climbing_photo_rear_glory_topo_classifier\climbing_classifier_data\topo"

# %%
def find_duplicate_images(folder_path):
    # Dictionary to store image hashes and their corresponding file paths
    hash_dict = defaultdict(list)

    # Iterate through all files in the specified folder
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # Join the folder path and file name
            file_path = os.path.join(root, file_name)

            # Try to open the image using Pillow
            try:
                img = Image.open(file_path)
            except Exception as e:
                # Skip files that are not valid images
                continue

            # Calculate the perceptual hash of the image
            img_hash = imagehash.average_hash(img)

            # Add the hash and file path to the dictionary
            hash_dict[img_hash].append(file_path)

    # Filter out hashes with only one file (not duplicates)
    duplicate_hashes = {k: v for k, v in hash_dict.items() if len(v) > 1}

    # Rename potential duplicate images with sequential integers
    duplicate_count = 0
    for hash_value, file_paths in duplicate_hashes.items():
        if len(file_paths) > 1:
            duplicate_count += 1
            for i, file_path in enumerate(file_paths):
                # Get the file extension (if any)
                _, file_extension = os.path.splitext(file_path)

                # Construct the new file name with the sequential integer
                new_name = f"potential_duplicate_{duplicate_count}_{i + 1}"

                # Construct the new file path
                new_path = os.path.join(os.path.dirname(file_path), new_name + file_extension)

                # Rename the file
                os.rename(file_path, new_path)

                print(f'Renamed: {file_path} to {new_path}')
    print(f"{duplicate_count} duplicates found")
    
find_duplicate_images(folder_path)

# %%
def delete_potential_duplicates(folder_path):
    # Iterate through all files in the specified folder
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_name_without_ext, file_extension = os.path.splitext(file_name)
            # Check if the file name matches the pattern "potential_duplicate_xxx_y"
            if file_name_without_ext.startswith("potential_duplicate_"):
                try:
                    # Extract the last number from the file name
                    last_number = int(file_name_without_ext.split("_")[-1])

                    # Check if the last number is greater than 1
                    if last_number > 1:
                        # Construct the file path
                        file_path = os.path.join(root, file_name)

                        # Delete the file
                        os.remove(file_path)

                        print(f'Deleted: {file_path}')
                except ValueError as e:
                    # Ignore files that don't match the expected pattern
                    print(e)
                    pass
                
delete_potential_duplicates(folder_path)
# %%
def rename_files_with_sequence(folder_path, base_string):
    # Get a list of all files in the specified folder
    files = os.listdir(folder_path)

    # Ensure the base string is not empty
    if not base_string:
        print("Error: Please provide a non-empty base string.")
        return

    # Rename each file with the provided string and a sequential integer
    for i, file_name in enumerate(files):
        # Get the file extension (if any)
        _, file_extension = os.path.splitext(file_name)

        # Construct the new file name with the base string and sequential integer
        new_name = f"{base_string}_{i + 1}"

        # Construct the new file path
        new_path = os.path.join(folder_path, new_name + file_extension)

        # Rename the file
        os.rename(os.path.join(folder_path, file_name), new_path)

        print(f'Renamed: {file_name} to {new_name}')

provided_string = input("Enter the base string for renaming: ")
rename_files_with_sequence(folder_path, provided_string)
# %%
