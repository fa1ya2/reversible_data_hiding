import os
import numpy as np
from src import image_preprocessing, performance_evaluation

def test_calculate_psnr():
    input_image_path = os.path.join('..', 'examples', 'sample_image.jpg')
    image1 = image_preprocessing.preprocess_image(input_image_path)
    image2 = np.copy(image1)

    psnr = performance_evaluation.calculate_psnr(image1, image2)
    assert psnr == float('inf'), "The PSNR of identical images should be infinite."

    image2 += 1
    psnr = performance_evaluation.calculate_psnr(image1, image2)
    assert psnr < float('inf'), "The PSNR of different images should be finite."

if __name__ == "__main__":
    test_calculate_psnr()
