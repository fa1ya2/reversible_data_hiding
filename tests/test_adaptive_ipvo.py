import os
import numpy as np
from src import image_preprocessing, adaptive_ipvo

def test_calculate_histogram():
    input_image_path = os.path.join('..', 'examples', 'sample_image.jpg')
    image = image_preprocessing.preprocess_image(input_image_path)
    histogram = adaptive_ipvo.calculate_histogram(image)

    assert np.sum(histogram) == image.size, "The histogram should represent the total number of pixels in the image."

def test_find_optimal_threshold(self):
    histogram = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
    optimal_threshold = adaptive_ipvo.find_optimal_threshold(histogram)
    self.assertIsNotNone(optimal_threshold)
    self.assertTrue(0 <= optimal_threshold < 256)

def test_adaptive_ipvo(self):
    input_image = np.array([
        [50, 10, 30],
        [20, 60, 40],
        [70, 80, 90]
    ], dtype=np.uint8)

    expected_reordered_image = np.array([
        [80, 70, 90],
        [60, 50, 20],
        [40, 30, 10]
    ], dtype=np.uint8)

    expected_binary_image = np.array([
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 0]
    ], dtype=np.uint8)

    reordered_image, binary_image = adaptive_ipvo(input_image)
    np.testing.assert_array_equal(reordered_image, expected_reordered_image)
    np.testing.assert_array_equal(binary_image, expected_binary_image)
