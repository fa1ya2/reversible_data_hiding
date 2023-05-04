import os
import numpy as np
from src import image_preprocessing, adaptive_ipvo, two_segment_pairwise_pee, data_embedding, data_extraction, performance_evaluation

def main():
    # File paths
    input_image_path = os.path.join('..', 'examples', 'sample_image.jpg')

    # Preprocess the image
    image = image_preprocessing.preprocess_image(input_image_path)

    # Perform adaptive IPVO
    reordered_image, binary_image = adaptive_ipvo.adaptive_ipvo(image)

    # Calculate two-segment pairwise PEE
    prediction_errors, two_segment_threshold = two_segment_pairwise_pee.two_segment_pairwise_pee(reordered_image)

    # Generate some random data to embed
    data = np.random.randint(0, 256, size=1000, dtype=np.uint8)

    # Embed the data
    embedded_image, binary_image = data_embedding.embed_data(reordered_image, binary_image, prediction_errors, two_segment_threshold, data)

    # Extract the data
    extracted_data = data_extraction.extract_data(reordered_image, embedded_image, binary_image, two_segment_threshold)

    # Restore the image
    restored_image = data_extraction.restore_image(embedded_image, binary_image)

    # Evaluate the performance
    capacity, efficiency, quality = performance_evaluation.evaluate_performance(image, embedded_image, restored_image, data)

    print(f"Payload capacity: {capacity}")
    print(f"Embedding efficiency: {efficiency}")
    print(f"Image quality (PSNR): {quality} dB")

if __name__ == "__main__":
    main()
