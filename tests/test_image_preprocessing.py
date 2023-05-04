import os
import numpy as np
import cv2
from src import image_preprocessing

def test_read_image():
    input_image_path = os.path.join('..', 'examples', 'sample_image.jpg')
    image = image_preprocessing.read_image(input_image_path)

    assert image is not None, "The image should not be None."

def test_resize_image():
    input_image_path = os.path.join('..', 'examples', 'sample_image.jpg')
    image = image_preprocessing.read_image(input_image_path)
    resized_image = image_preprocessing.resize_image(image, 200, 200)

    assert resized_image.shape == (200, 200), "The resized image should have the specified dimensions."

def test_normalize_image():
    image = np.array([[260, -10], [300, 500]], dtype=np.int32)
    normalized_image = image_preprocessing.normalize_image(image)

    assert np.min(normalized_image) >= 0 and np.max(normalized_image) <= 255, "The normalized image should have pixel values between 0 and 255."

if __name__ == "__main__":
    test_read_image()
    test_resize_image()
    test_normalize_image()
