import cv2
from skimage import exposure


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def adaptive_histogram_equalization(image):
    adaptive_histogram_equalizer = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(4,4))
    return adaptive_histogram_equalizer.apply(image)


def image_adjust(image):
    return exposure.rescale_intensity(image)


def histogram_equalization(image):
    return cv2.equalizeHist(image)


def contrast_normalization(image):
    blurred = cv2.GaussianBlur(image, (3,3), 0)
    return cv2.Laplacian(blurred, cv2.CV_8U, 3)


def flatten(image):
    return image.ravel()


def run(train_input):
    preprocessed_images = []

    for image in train_input:
        preprocessed_image = grayscale(image)
        preprocessed_image = histogram_equalization(preprocessed_image)
        preprocessed_image = flatten(preprocessed_image)
        preprocessed_images.append(preprocessed_image)

    return preprocessed_images