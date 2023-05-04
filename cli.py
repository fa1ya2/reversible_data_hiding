import argparse
import os
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
    parser.add_argument("input_image", help="Path to the input image file")
    parser.add_argument("-d", "--data", help="Data to be embedded", required=True)
    parser.add_argument("-o", "--output_dir", help="Directory to store the output images and extracted data", default="output/")
    args = parser.parse_args()

    input_image_path = args.input_image
    data = args.data.encode()
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load image and generate histogram
    image = load_image(input_image_path)
    grayscale_image = convert_to_grayscale(image)
    histogram = generate_histogram(grayscale_image)

    # Adaptive IPVO
    threshold = find_optimal_threshold(histogram)

    # Two-segment Pairwise PEE
    optimized_histogram, two_segment_threshold = two_segment_pairwise_pee(grayscale_image)

    # Calculate height and width of the grayscale image
    height, width = grayscale_image.shape

    # Convert secret data to binary representation
    binary_data = data_to_binary(data, width, height)
    # print("Binary data:", binary_data[:len(data) * 8])

    # Data embedding
    embedded_image = embed_data(grayscale_image, optimized_histogram, two_segment_threshold, binary_data)

    # Data extraction
    # print("Binary data before extraction:", binary_data[:len(data) * 8])
    extracted_data, restored_image = extract_data(grayscale_image, embedded_image, optimized_histogram, two_segment_threshold, binary_data)
    # print("Extracted data:", extracted_data[:len(data) * 8])

    # Save output images and extracted data
    save_image(os.path.join(output_dir, "embedded_image.png"), embedded_image)
    save_image(os.path.join(output_dir, "restored_image.png"), restored_image)

    with open(os.path.join(output_dir, "extracted_data_encrypted.txt"), "w") as f:
        f.write(extracted_data.tobytes().decode(errors="ignore"))

    decoded_message = "".join(chr(byte) for byte in extracted_data)
    # print("Decoded message:", decoded_message)

    with open(os.path.join(output_dir, "extracted_data_decrypted.txt"), "w") as m:
        m.write(decoded_message)

    # Performance evaluation
    payload_capacity_value = payload_capacity(embedded_image)
    embedding_efficiency_value = embedding_efficiency(embedded_image, grayscale_image)
    psnr_value = psnr(grayscale_image, restored_image)

    print("Payload Capacity:", payload_capacity_value)
    print("Embedding Efficiency:", embedding_efficiency_value)
    print("PSNR:", psnr_value)


if __name__ == "__main__":
    main()
