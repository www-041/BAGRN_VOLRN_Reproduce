"""Unit tests for the opt-in KLT/TPS registration backend."""


def test_opencv_dependency_is_available():
    import cv2

    assert cv2.__version__
