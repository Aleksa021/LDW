"""Dataset calibration loading.

The Jiqing dataset ships an OpenCV-YAML `parameters.yml` with a 3x3 intrinsic
matrix K and a 4x1 distortion vector D. The 4x1 shape is OpenCV's *fisheye*
convention (k1, k2, k3, k4), confirmed by the implausibly large coefficient
under the pinhole interpretation.

Undistortion itself lives in the detector (ldw/detectors/cv_detector.py), which
rectifies for processing and re-distorts its output back to the original image.
This module only reads the calibration so the runner can hand (K, D) to the
detector.
"""

from typing import Tuple

import cv2
import numpy as np


def load_parameters(yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read K (3x3) and D (4x1) from an OpenCV-YAML parameters file."""
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Could not open calibration YAML: {yaml_path}")
    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    fs.release()
    if K is None or D is None:
        raise RuntimeError(f"K or D missing in {yaml_path}")
    return K.astype(np.float64), D.reshape(-1, 1).astype(np.float64)
