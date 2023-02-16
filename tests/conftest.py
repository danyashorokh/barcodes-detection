
import os
import pytest

import cv2
import numpy as np

from src.object_detection.wrapper import ObjectDetector

from config_tests import Config

TESTS_DIR = os.path.dirname(__file__)


@pytest.fixture(scope='session')
def dummy_input():
    return np.random.rand(*Config.image_size, 3)


@pytest.fixture(scope='session')
def sample_image_np():
    img = cv2.imread(os.path.join(TESTS_DIR, 'fixtures', 'images',
                     '0b56af7e-386c-410a-8f46-74350f755d77--ru.4c7208d1-cba4-4539-967d-1c87ad0f6d2e.jpg'))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope='session')
def model_wrapper():
    return ObjectDetector(weights=Config.weights, device=Config.device,
                          threshold=Config.object_threshold)
