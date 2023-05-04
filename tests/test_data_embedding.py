import os
import numpy as np
from src import image_preprocessing, adaptive_ipvo, two_segment_pairwise_pee, data_embedding

def test_embed_data():
    input_image_path = os.path.join('..', 'examples', 'sample_image.jpg')
    image = image_preprocessing.preprocess_image(input_image_path)
    reordered_image, binary_image = adaptive_ipvo.adaptive_ipvo(image)
    prediction_errors, two_segment_threshold = two_segment_pairwise_pee.two_segment_pairwise_pee(reordered_image)

    data = np.random.randint(0, 256, size=1000, dtype=np.uint8)
    embedded_image, updated_binary_image = data_embedding.embed_data(reordered_image, binary_image, prediction_errors, two_segment_threshold, data)

    assert embedded_image.shape == image.shape, "The embedded image should have the same shape as the input image."

if __name__ == "__main__":
    test_embed_data()
