import pytest
from head_pose.src.demo import head_pose
import cv2


def test_regression():
    status = 0
    image1 = head_pose()
    image2 = cv2.imread("reference", 0)
    if image1.shape == image2.shape and not (np.bitwise_xor(image1, image2).any()):
        status = 1
    else:
        status = 0
    assert status == 1
