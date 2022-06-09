import pytest
from src.demo import head_pose
import cv2
import pkg_resources
import numpy as np


def test_regression():
    status = 0
    image1 = head_pose()
    path_to_file = pkg_resources.resource_filename("tests", "reference.jpeg")
    image2 = cv2.imread(path_to_file, 0)
    if image1.shape == image2.shape and not (np.bitwise_xor(image1, image2).any()):
        status = 1
    else:
        status = 0
    assert status == 1
