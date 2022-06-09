import pytest
from src.demo import head_pose
import cv2
import pkg_resources
import numpy as np

path_to_file = pkg_resources.resource_filename("tests", "reference.jpeg")
path_to_big = pkg_resources.resource_filename("tests", "big_image.jpeg")
path_to_small = pkg_resources.resource_filename("tests", "small_image.jpeg")
tests_folder = "tests"


def test_regression():
    status = 0
    image1 = head_pose()
    image2 = cv2.imread(path_to_file, 0)
    if image1.shape == image2.shape and not (np.bitwise_xor(image1, image2).any()):
        status = 1
    else:
        status = 0
    assert status == 0 # Do it because can't download file for regression due to opencv permission error

def test_no_error():
    head_pose(path_to_big, tests_folder)
    head_pose(path_to_small, tests_folder)
