import numpy as np

def calculate_prediction_errors(image):
    height, width = image.shape
    prediction_errors = np.zeros((height - 1, width - 1), dtype=np.int16)

    for y in range(height - 1):
        for x in range(width - 1):
            prediction_errors[y, x] = int(image[y, x]) - int(image[y + 1, x + 1])

    return prediction_errors

def calculate_two_segment_threshold(prediction_errors):
    #512
    histogram = np.zeros(512, dtype=np.int32)
    for error in prediction_errors.ravel():
        histogram[error + 255] += 1

    max_frequency = 0
    two_segment_threshold = 0

    #512
    for i in range(512):
        if histogram[i] > max_frequency:
            max_frequency = histogram[i]
            two_segment_threshold = i - 255

    return two_segment_threshold

def two_segment_pairwise_pee(image):
    prediction_errors = calculate_prediction_errors(image)
    two_segment_threshold = calculate_two_segment_threshold(prediction_errors)

    return prediction_errors, two_segment_threshold
