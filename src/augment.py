import numpy as np
import cv2
import pickle
import tqdm

TRAIN_FILE = '../data/train.pickle'

TRANSLATION_RANGE = (-2, 2)
SCALE_RANGE = (0.9, 1.1)
ROTATION_RANGE = (-15, 15)

DEFAULT_TRANSFORMS = [[-2, 0], [0, -2], [2, 0], [0, 2]]
DEFAULT_SCALES = list(SCALE_RANGE)
DEFAULT_ANGLES = list(ROTATION_RANGE)

SAMPLES = 4
STOCHASTIC_TRANSFORMS = np.split(np.random.random_integers(low=TRANSLATION_RANGE[0], high=TRANSLATION_RANGE[1], size=SAMPLES), 2)
STOCHASTIC_SCALES = np.random.uniform(low=SCALE_RANGE[0], high=SCALE_RANGE[1], size=SAMPLES)
STOCHASTIC_ANGLES = np.random.random_integers(low=ROTATION_RANGE[0], high=ROTATION_RANGE[1], size=SAMPLES)


def translate(image, transforms=DEFAULT_TRANSFORMS):
    translated = []

    for transform in transforms:
        m = np.float32([[1, 0, transform[0]], [0, 1, transform[1]]])
        translated.append(cv2.warpAffine(image, m, dsize=(image.shape[0], image.shape[1])))

    return translated


def scale(image, scales=DEFAULT_SCALES):
    scaled = []

    for random_scale in scales:
        scaled_image = cv2.resize(image, None, fx=random_scale, fy=random_scale)
        scaled_image = cv2.resize(scaled_image, dsize=(image.shape[0], image.shape[1]))
        scaled.append(scaled_image)
    return scaled


def rotate(image, angles=DEFAULT_ANGLES):
    rotated = []

    for angle in angles:
        center = tuple(np.array(image.shape[:2]) / 2)
        R = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, R, image.shape[:2], flags=cv2.INTER_CUBIC)
        rotated.append(rotated_image)

    return rotated


def deterministic(train_input, train_labels):
    augmented = { 'features': [], 'labels': [] }

    for image, label in tqdm.tqdm(zip(train_input, train_labels), desc='Augmenting', total=len(train_input)):
        augmented['features'].append(image)
        augmented['labels'].append(label)
        for _image in translate(image):
            augmented['features'].append(_image)
            augmented['labels'].append(label)
        for _image in scale(image):
            augmented['features'].append(_image)
            augmented['labels'].append(label)
        for _image in rotate(image):
            augmented['features'].append(_image)
            augmented['labels'].append(label)

    return augmented


def stochastic(train_input, train_label):
    augmented = []

    for image, label in zip(train_input, train_label):
        augmented.append(translate(image, transforms=STOCHASTIC_TRANSFORMS))
        augmented.append(scale(image, scales=STOCHASTIC_SCALES))
        augmented.append(rotate(image, angles=STOCHASTIC_ANGLES))

    return augmented


def run(file='../data/augmented_train.pickle'):
    with open(TRAIN_FILE, mode='rb') as f:
        train = pickle.load(f)

    augmented = deterministic(train['features'], train['labels'])

    pickle.dump(augmented, open(file, 'wb'))

run()