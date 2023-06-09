
from copy import deepcopy
import pytest

import numpy as np


@pytest.mark.skip(reason='в разработке')
def test_model_empty_tensor(model_wrapper, dummy_input):
    boxes, scores, labels_ids = model_wrapper.detect(dummy_input)
    assert not scores, scores


def test_model_np_image(model_wrapper, sample_image_np):
    boxes, scores, labels_ids = model_wrapper.detect(sample_image_np)
    assert scores, scores


def test_correct_probs(model_wrapper, dummy_input, sample_image_np):

    boxes, scores, labels_ids = model_wrapper.detect(dummy_input)

    for score in scores:
        assert score <= 1, score
        assert score >= 0, score

    for labels_id in labels_ids:
        assert labels_id in model_wrapper.classes.keys()


def test_predict_dont_mutate_orig_image(model_wrapper, sample_image_np):
    initial_image = deepcopy(sample_image_np)
    model_wrapper.detect(sample_image_np)
    assert np.allclose(initial_image, sample_image_np)
