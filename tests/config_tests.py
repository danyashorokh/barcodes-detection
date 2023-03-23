
import os

TESTS_DIR = os.path.dirname(__file__)


class Config:
    weights = os.path.join(TESTS_DIR, '..', 'weights', 'best_openvino_model')
    device = 'cpu'
    image_size = (1280, 720)
    object_threshold = 0.4
