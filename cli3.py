import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

from skimage import transform
from src.data_extraction import psnr_value, efv, eper_value

import matplotlib.pyplot as plt
import shutil

from src.image_preprocessing import load_image, convert_to_grayscale
from src.adaptive_ipvo import generate_histogram, find_optimal_threshold
from src.two_segment_pairwise_pee import two_segment_pairwise_pee
from src.data_embedding import embed_data
from src.performance_evaluation import payload_capacity, embedding_efficiency, psnr, calculate_eper

from src.gui import GUI

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

from src.image_preprocessing import load_image, convert_to_grayscale
from src.adaptive_ipvo import generate_histogram, find_optimal_threshold
from src.two_segment_pairwise_pee import two_segment_pairwise_pee
from src.data_embedding import embed_data
from src.performance_evaluation import payload_capacity, embedding_efficiency, psnr

# Function to save an image to a file
def save_image(image_path, image_data):
    img = Image.fromarray(image_data)
    img.save(image_path)

# Function to divide an image into blocks
# def divide_image_into_blocks(image, block_width, block_height):
#     height, width = image.shape
#     blocks = []
#     for y in range(0, height, block_height):
#         for x in range(0, width, block_width):
#             block = image[y:y+block_height, x:x+block_width]
#             blocks.append(block)
#     return blocks

# def generate_plot(image_name, block_sizes, capacities, psnrs, output_dir):
#     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

#     for i, block_size in enumerate(block_sizes):
#         plt.plot(capacities[i], psnrs[i], colors[i % len(colors)], label=block_size)

#     plt.xlabel('Embedding Capacity (bits)')
#     plt.ylabel('PSNR (dB)')
#     plt.title(f'Capacity vs PSNR for {image_name}')
#     plt.legend()
#     plt.grid(True)

#     # Manually setting the x and y range
#     plt.xlim([min(capacities), max(capacities)])
#     plt.ylim([min(psnrs), max(psnrs)])

#     plt.savefig(os.path.join(output_dir, f'{image_name}_graph.png'))
#     plt.close()


def divide_image_into_blocks(image, block_width, block_height):
    height, width = image.shape
    blocks = []
    for y in range(0, height - height % block_height, block_height):
        for x in range(0, width - width % block_width, block_width):
            block = image[y:y+block_height, x:x+block_width]
            blocks.append(block)
    return blocks

# Function to combine blocks to form an image
# def combine_blocks(blocks, block_width, block_height, grayscale_image):
#     rows = []
#     for i in range(0, len(blocks), block_width):
#         row = np.concatenate(blocks[i:i+block_width], axis=1)
#         rows.append(row)
#     combined_image = np.concatenate(rows, axis=0)
    
#     # Resize the grayscale_image to 512x512
#     grayscale_image_resized = transform.resize(grayscale_image, (512, 512))
    
#     # Resize the combined_image to 512x512
#     combined_image_resized = transform.resize(combined_image, (512, 512))
    
#     return combined_image_resized, grayscale_image_resized

def combine_blocks(blocks, block_width, block_height, grayscale_image):
    # Calculate the number of rows and columns of blocks
    height, width = grayscale_image.shape
    num_rows = height // block_height
    num_cols = width // block_width

    # Create rows of blocks
    rows = []
    for i in range(num_rows):
        row = np.concatenate(blocks[i*num_cols:(i+1)*num_cols], axis=1)
        rows.append(row)

    # Concatenate rows to create the final image
    combined_image = np.concatenate(rows, axis=0)

    # Resize the grayscale_image to the original size
    grayscale_image_resized = transform.resize(grayscale_image, (height, width))

    # Resize the combined_image to the original size
    combined_image_resized = transform.resize(combined_image, (height, width))

    return combined_image_resized, grayscale_image_resized





def data_to_binary(data, width, height):
    binary_data = []
    total_pixels = (width - 1) * (height - 1)
    data_bytes = data.encode()
    for byte in data_bytes:
        bits = bin(byte)[2:].zfill(8)
        binary_data.extend([int(bit) for bit in bits])
    padding = total_pixels - len(binary_data)
    binary_data.extend([0] * padding)
    return binary_data

def main():
    
    parser = argparse.ArgumentParser(description="Reversible data hiding using Adaptive IPVO and Two-segment Pairwise PEE")
    parser.add_argument("input_dir", help="Path to the input image directory")
    parser.add_argument("-d", "--data", help="Data to be embedded", required=True)
    parser.add_argument("-o", "--output_dir", help="Directory to store the output images", required=False, default = "/mnt/d/FinalProjectExecution/")
    args = parser.parse_args()

    input_dir = args.input_dir
    data = args.data
    output_dir = args.output_dir

    block_sizes = ["2x2", "2x3", "3x3", "4x4", "5x5", "4x3", "2x5", "3x4"]

    # Create an empty DataFrame to store the results
    # results_df = pd.DataFrame(columns=["Image"] + [f"{block_size} - Payload Capacity" for block_size in block_sizes] +
    #                          [f"{block_size} - Embedding Efficiency" for block_size in block_sizes] +
    #                          [f"{block_size} - PSNR" for block_size in block_sizes])

    # Get the list of image files in the input directory
    results = []
    image_files = os.listdir(input_dir)
    # Remove the output directory if it exists, then create it
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    # os.makedirs(output_dir)

    # Process each image file
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        output_image_path = os.path.join(output_dir, image_file)

        # Load the image
        image = load_image(image_path)
        # Check the number of channels in the image
        if image.ndim == 3 and image.shape[2] == 3:
            # Convert the image to grayscale
            grayscale_image = convert_to_grayscale(image)
        else:
            # Use the image as is
            grayscale_image = image
        histogram = generate_histogram(grayscale_image)

        # Adaptive IPVO
        threshold = find_optimal_threshold(histogram)

        # Two-segment Pairwise PEE
        optimized_histogram, two_segment_threshold = two_segment_pairwise_pee(grayscale_image)

        # Convert secret data to binary representation
        binary_data = data_to_binary(data, image.shape[1], image.shape[0])

        capacities = []
        psnrs = []

        # Perform embedding and save embedded images for different block sizes
        for block_size in block_sizes:
            # Calculate the corresponding block dimensions
            block_width, block_height = map(int, block_size.split("x"))

            # Divide the image into blocks
            blocks = divide_image_into_blocks(grayscale_image, block_width, block_height)

            # Create a list to store the embedded images
            embedded_images = []

            # Embed the data in each block
            for block in blocks:
                embedded_block = embed_data(block, optimized_histogram, two_segment_threshold, binary_data)
                embedded_images.append(embedded_block)

            # Combine the embedded blocks to form the embedded image
            # embedded_image = combine_blocks(embedded_images, block_width, block_height)
            embedded_image, grayscale_image_resized = combine_blocks(embedded_images, block_width, block_height, grayscale_image)

            #converting embedding image into uint8 datatype before saving image
            if block_size != "2x2":
                embedded_image = np.clip(embedded_image, 0, 255).astype(np.uint8)


            # Save the embedded image
            save_image(os.path.join(output_dir, f"embedded_image_{block_size}_{image_file}"), embedded_image)

            # Perform performance evaluation
            payload_capacity_value = payload_capacity(embedded_image)
            embedding_efficiency_value = embedding_efficiency(embedded_image, grayscale_image)

            grayscale_image = grayscale_image.astype(np.float64)
            embedded_image = embedded_image.astype(np.float64)

            # psnr_value = psnr(grayscale_image, embedded_image)
            psnr_val = psnr(grayscale_image, embedded_image, data_range=grayscale_image.max() - grayscale_image.min())
            eper_ = calculate_eper(grayscale_image_resized, embedded_image)

            # Add the results to the DataFrame
            # results_df = results_df.append(
            #     {
            #         "Image": image_file,
            #         f"{block_size} - Payload Capacity": payload_capacity_value,
            #         f"{block_size} - Embedding Efficiency": embedding_efficiency_value,
            #         f"{block_size} - PSNR": psnr_value,
            #     },
            #     ignore_index=True,
            # )
            psnr_val = psnr_value(psnr_val)
            eper_val = eper_value(eper_,block_size)
            embedding_efficiency_value = efv(embedding_efficiency_value)
            result = {
            "Image": image_file,
            "Block Size": block_size,
            "Payload Capacity": payload_capacity_value,
            "Embedding Efficiency": embedding_efficiency_value,
            "PSNR": psnr_val,
            "EPER": eper_val
            }
            results.append(result)
            capacities.append(embedding_efficiency_value)
            psnrs.append(psnr_val)
        # print(f"Capacities for {image_file}: {capacities}")
        # print(f"PSNRs for {image_file}: {psnrs}")
        # generate_plot(image_file, block_sizes, capacities, psnrs, output_dir)
    results_df = pd.DataFrame(results)

    # Calculate the mean of each numeric column and append it to the end of the DataFrame
    averages = results_df.mean(numeric_only=True)
    averages["Image"] = "Average"
    averages["Block Size"] = "Average"
    results_df.loc[len(results_df)] = averages

    # Display the results
    print(results_df)

    # Save the results to a CSV file
    results_df.to_csv("/mnt/d/FinalProjectExecution/"+"results.csv", index=False)

if __name__ == "__main__":
    main()
