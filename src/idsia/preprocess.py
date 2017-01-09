import cv2
from skimage import exposure

COLOR_SPACE = cv2.COLOR_RGB2LAB
INVERSE_COLOR_SPACE = cv2.COLOR_LAB2RGB
INTENSITY_COMPONENT = 1  # L


def adaptive_histogram_equalization(image):
    image = cv2.cvtColor(image, COLOR_SPACE)
    x, y, z = cv2.split(image)

    adaptive_histogram_equalizer = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(4,4))

    if INTENSITY_COMPONENT == 1:
        x = adaptive_histogram_equalizer.apply(x)
    elif INTENSITY_COMPONENT == 2:
        y = adaptive_histogram_equalizer.apply(y)
    elif INTENSITY_COMPONENT == 3:
        z = adaptive_histogram_equalizer.apply(z)

    return cv2.cvtColor(cv2.merge((x, y, z)), INVERSE_COLOR_SPACE)


def image_adjust(image):
    image = cv2.cvtColor(image, COLOR_SPACE)
    x, y, z = cv2.split(image)

    if INTENSITY_COMPONENT == 1:
        x = exposure.rescale_intensity(x)
    elif INTENSITY_COMPONENT == 2:
        y = exposure.rescale_intensity(y)
    elif INTENSITY_COMPONENT == 3:
        z = exposure.rescale_intensity(z)

    return cv2.cvtColor(cv2.merge((x, y, z)), INVERSE_COLOR_SPACE)


def histogram_equalization(image):
    image = cv2.cvtColor(image, COLOR_SPACE)
    x, y, z = cv2.split(image)

    if INTENSITY_COMPONENT == 1:
        x = cv2.equalizeHist(x)
    elif INTENSITY_COMPONENT == 2:
        y = cv2.equalizeHist(y)
    elif INTENSITY_COMPONENT == 3:
        z = cv2.equalizeHist(z)

    return cv2.cvtColor(cv2.merge((x, y, z)), INVERSE_COLOR_SPACE)

def preprocess(train_input):
    preprocessed_images = []

    for image in train_input:
        preprocessed_image = image_adjust(image)
        preprocessed_images.append(preprocessed_image)

    return preprocessed_images