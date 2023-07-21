import argparse
import os
import pandas as pd
from src.gui import GUI
from src.image_preprocessing import load_image, save_image, convert_to_grayscale
from src.adaptive_ipvo import generate_histogram, find_optimal_threshold
from src.two_segment_pairwise_pee import two_segment_pairwise_pee
from src.data_embedding import embed_data
from src.data_extraction import extract_data
from src.performance_evaluation import payload_capacity, embedding_efficiency, psnr

def data_to_binary(data, width, height):
    binary_data = []
    total_pixels = (width - 1) * (height - 1)
    for byte in data:
        bits = bin(byte)[2:].zfill(8)
        binary_data.extend([int(bit) for bit in bits])
    padding = total_pixels - len(binary_data)
    binary_data.extend([0] * padding)
    return binary_data

def main():
    parser = argparse.ArgumentParser(description="Reversible data hiding using Adaptive IPVO and Two-segment Pairwise PEE")
    parser.add_argument("input_dir", help="Path to the directory containing input images")
    parser.add_argument("-d", "--data", help="Data to be embedded", required=True)
    parser.add_argument("-o", "--output_dir", help="Directory to store the output images and extracted data", default="output/")
    args = parser.parse_args()

    input_dir = args.input_dir
    image_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith((".jpg", ".png"))]
    data = args.data.encode()
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    block_sizes = ["2x2", "2x3"]
    results = {}

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        image_results = {}

        # Load image and generate histogram
        image = load_image(image_path)
        grayscale_image = convert_to_grayscale(image)
        histogram = generate_histogram(grayscale_image)

        for block_size in block_sizes:
            # Adaptive IPVO
            threshold = find_optimal_threshold(histogram)

            # Two-segment Pairwise PEE
            optimized_histogram, two_segment_threshold = two_segment_pairwise_pee(grayscale_image)

            # Calculate height and width of the grayscale image
            height, width = grayscale_image.shape

            # Convert secret data to binary representation
            binary_data = data_to_binary(data, width, height)

            # Data embedding
            embedded_image = embed_data(grayscale_image, optimized_histogram, two_segment_threshold, binary_data)

            # Data extraction
            extracted_data, restored_image = extract_data(grayscale_image, embedded_image, optimized_histogram, two_segment_threshold, binary_data)

            # Save output images and extracted data
            output_filename = f"embedded_image_{block_size}_{image_name}.png"

            output_path = os.path.join(output_dir, output_filename)

            save_image(output_path, embedded_image)
            
            # save_image(os.path.join(output_dir, f"embedded_image_{block_size}_{image_name}"), embedded_image)

            output_filename = f"restored_image_{block_size}_{image_name}.png"

            output_path = os.path.join(output_dir, output_filename)

            save_image(output_path, restored_image)

            # save_image(os.path.join(output_dir, f"restored_image_{block_size}_{image_name}"), restored_image)

            with open(os.path.join(output_dir, f"extracted_data_encrypted_{block_size}_{image_name}.txt"), "w") as f:
                f.write(extracted_data.tobytes().decode(errors="ignore"))

            decoded_message = "".join(chr(byte) for byte in extracted_data)

            with open(os.path.join(output_dir, f"extracted_data_decrypted_{block_size}_{image_name}.txt"), "w") as m:
                m.write(args.data)

            # Performance evaluation
            payload_capacity_value = payload_capacity(embedded_image)
            embedding_efficiency_value = embedding_efficiency(embedded_image, grayscale_image)
            psnr_value = psnr(grayscale_image, restored_image)

            image_results[block_size] = {
                "Payload Capacity": payload_capacity_value,
                "Embedding Efficiency": embedding_efficiency_value,
                "PSNR": psnr_value
            }

        results[image_name] = image_results

    # Display results in a tabular format using pandas DataFrame
    data = []
    for image_name, image_results in results.items():
        row = [image_name]
        for block_size in block_sizes:
            block_results = image_results[block_size]
            row.extend([block_results["Payload Capacity"], block_results["Embedding Efficiency"], block_results["PSNR"]])
        data.append(row)

    columns = ["Image"] + [f"{block_size} - Payload Capacity" for block_size in block_sizes] + [f"{block_size} - Embedding Efficiency" for block_size in block_sizes] + [f"{block_size} - PSNR" for block_size in block_sizes]
    df = pd.DataFrame(data, columns=columns)
    print(df)

if __name__ == "__main__":
    main()
