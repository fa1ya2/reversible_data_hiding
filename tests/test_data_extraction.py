import os
import numpy as np
from src import image_preprocessing, adaptive_ipvo, two_segment_pairwise_pee, data_embedding, data_extraction

def test_extract_data():
    input_image_path = os.path.join('..', 'examples', 'sample_image.jpg')
    image = image_preprocessing.preprocess_image(input_image_path)
    reordered_image, binary_image = adaptive_ipvo.adaptive_ipvo(image)
    prediction_errors, two_segment_threshold = two_segment_pairwise_pee.two_segment_pairwise_pee(reordered_image)

    data = np.random.randint(0, 256, size=1000, dtype=np.uint8)
    embedded_image, updated_binary_image = data_embedding.embed_data(reordered_image, binary_image, prediction_errors, two_segment_threshold, data)

    extracted_data = data_extraction.extract_data(reordered_image, embedded_image, updated_binary_image, two_segment_threshold)
    assert np.array_equal(data, extracted_data), "The extracted data should match the original data."

def test_restore_image():
    input_image_path = os.path.join('..', 'examples', 'sample_image.jpg')
    image = image_preprocessing.preprocess_image(input_image_path)
    reordered_image, binary_image = adaptive_ipvo.adaptive_ipvo(image)
    prediction_errors, two_segment_threshold = two_segment_pairwise_pee.two_segment_pairwise_pee(reordered_image)

    data = np.random.randint(0, 256, size=1000, dtype=np.uint8)
    embedded_image, updated_binary_image = data_embedding.embed_data(reordered_image, binary_image, prediction_errors, two_segment_threshold, data)

    restored_image = data_extraction.restore_image(embedded_image, updated_binary_image, two_segment_threshold)
    assert np.array_equal(reordered_image, restored_image), "The restored image should match the original reordered image."

if __name__ == "__main__":
    test_extract_data()
    test_restore_image()
