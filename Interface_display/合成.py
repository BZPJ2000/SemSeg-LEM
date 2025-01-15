import os
import numpy as np
from PIL import Image

def merge_images_by_group(input_folder, output_folder, blacken_height=22):
    """
    Merge images in the input folder by their grouped prefixes and save the merged results in the output folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save merged output images.
        blacken_height (int): Height in pixels to blacken at the top of the merged images (default is 22).
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all files starting with "marked_" and ending with ".png"
    all_files = os.listdir(input_folder)
    marked_files = [f for f in all_files if f.startswith('marked_') and f.endswith('.png')]

    # Group images by their prefixes (e.g., "marked_00001_")
    grouped_images = {}
    for file_name in marked_files:
        # Extract prefix, e.g., "marked_00001_"
        prefix = '_'.join(file_name.split('_')[:3]) + '_'
        if prefix not in grouped_images:
            grouped_images[prefix] = []
        grouped_images[prefix].append(file_name)

    # Process each group of images
    for prefix, image_names in grouped_images.items():
        merged_image = None

        for name in image_names:
            image_path = os.path.join(input_folder, name)
            image = np.array(Image.open(image_path))

            # Initialize merged_image for the first image in the group
            if merged_image is None:
                merged_image = np.zeros_like(image)

            # Merge current image into merged_image
            merged_image = np.maximum(merged_image, image)

        # Blacken the top region of the merged image
        merged_image[:blacken_height, :] = 0

        # Save the merged image
        output_filename = prefix + 'merged.png'
        output_path = os.path.join(output_folder, output_filename)
        Image.fromarray(merged_image).save(output_path)

        print(f'Merged image saved to: {output_path}')

# input_folder = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_B'
# output_folder = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_C'
#
# merge_images_by_group(input_folder, output_folder, blacken_height=23)
