"""Dataset calibration + fisheye rectification helpers.

The Jiqing dataset ships an OpenCV-YAML `parameters.yml` with a 3x3 intrinsic
matrix K and a 4x1 distortion vector D. The 4x1 shape is OpenCV's *fisheye*
convention (k1, k2, k3, k4), confirmed by the implausibly large 4th coefficient
under the pinhole interpretation.

These helpers undistort both the frame (for the CV detector) and the GT points
with the *same* model + camera matrix, so predictions and GT live in one space.
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


class FisheyeRectifier:
    """Rectifies frames and points with the fisheye model, sharing one Knew.

    Knew defaults to K (same size, may crop FOV). Pass `balance` to widen the
    retained field of view via estimateNewCameraMatrixForUndistortRectify.
    """

    def __init__(self, K: np.ndarray, D: np.ndarray, image_size: Tuple[int, int],
                 balance: float = 0.0, use_knew: bool = True):
        self.K = np.asarray(K, dtype=np.float64)
        self.D = np.asarray(D, dtype=np.float64).reshape(-1, 1)
        self.size = (int(image_size[0]), int(image_size[1]))  # (W, H)

        if use_knew:
            self.Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.D, self.size, np.eye(3), balance=balance)
        else:
            self.Knew = self.K.copy()

        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.Knew, self.size, cv2.CV_16SC2)

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """pts: (N, 2) distorted pixel coords -> (N, 2) rectified pixel coords."""
        if len(pts) == 0:
            return pts.astype(np.float32)
        src = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        dst = cv2.fisheye.undistortPoints(src, self.K, self.D, P=self.Knew)
        return dst.reshape(-1, 2).astype(np.float32)
