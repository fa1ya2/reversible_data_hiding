import os
import numpy as np
from src import image_preprocessing, adaptive_ipvo, two_segment_pairwise_pee

def test_calculate_prediction_errors():
    input_image_path = os.path.join('..', 'examples', 'sample_image.jpg')
    image = image_preprocessing.preprocess_image(input_image_path)
    reordered_image, binary_image = adaptive_ipvo.adaptive_ipvo(image)

    prediction_errors = two_segment_pairwise_pee.calculate_prediction_errors(reordered_image)
    assert prediction_errors.shape == image.shape, "Prediction errors should have the same shape as the input image."

def test_calculate_two_segment_threshold():
    prediction_errors = np.array([-5, -1, 0, 2, 5, 8, -3])
    two_segment_threshold = two_segment_pairwise_pee.calculate_two_segment_threshold(prediction_errors)

    assert isinstance(two_segment_threshold, int), "The two-segment threshold should be an integer."

if __name__ == "__main__":
    test_calculate_prediction_errors()
    test_calculate_two_segment_threshold()
