import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

def payload_capacity(data):
    return len(data)

def embedding_efficiency(embedded_image, original_image):
    diff = np.abs(embedded_image.astype(np.int64) - original_image.astype(np.int64))
    return np.sum(diff) / np.sum(np.abs(original_image))

def image_quality(original_image, restored_image):
    return psnr(original_image, restored_image)

def evaluate_performance(original_image, embedded_image, restored_image, data):
    capacity = payload_capacity(data)
    efficiency = embedding_efficiency(original_image, embedded_image)
    quality = image_quality(original_image, restored_image)

    return capacity, efficiency, quality
