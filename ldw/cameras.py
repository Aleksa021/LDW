"""Camera presets — one source of truth for per-camera facts.

The project targets exactly two cameras: the Jiqing Expressway dashcam (fisheye)
and the custom e-con camera (pinhole, calibrated later). Everything camera-
specific — intrinsics K, distortion D + model, the CV perspective ROI, and the
BEV homography H — lives here so detectors, the LDW system, and the eval runner
all read the same definition instead of scattering magic numbers.

Lane points flow as RAW original-image pixels everywhere (see detectors/base.py);
undistortion happens where it's needed (inside the CV detector for processing,
inside the LDW system before the homography) using a camera's K/D/model.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CameraConfig:
    name: str
    image_size: Tuple[int, int]          # (W, H)
    distortion_model: str                # "fisheye" | "pinhole"
    K: Optional[np.ndarray]              # 3x3 intrinsics
    D: Optional[np.ndarray]              # distortion (fisheye 4x1; pinhole k1,k2,p1,p2[,k3])
    cv_roi: Optional[dict] = None        # CV perspective trapezoid (fractions of W,H)
    H: Optional[np.ndarray] = None       # BEV homography; None until calibrated

    @property
    def calibration(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return None if self.K is None else (self.K, self.D)


# CV perspective trapezoid for the Jiqing camera, in the undistorted view where
# CV detection happens. Fractions of (W, H), order BL, BR, TR, TL. Tuned from the
# undistorted ego-lane geometry.
_JIQING_CV_ROI = {
    "src": [(0.22, 0.86), (0.78, 0.86), (0.575, 0.60), (0.445, 0.60)],
    "dst": [(0.2266, 1.0), (0.7734, 1.0), (0.7734, 0.0), (0.2266, 0.0)],
}

# Jiqing Expressway dashcam — values from the dataset's parameters.yml (fisheye).
JIQING = CameraConfig(
    name="jiqing",
    image_size=(1920, 1080),
    distortion_model="fisheye",
    K=np.array([[1107.4698151168973, 0.0, 995.52519781648857],
                [0.0, 1124.1674862922000, 546.29877445023601],
                [0.0, 0.0, 1.0]], dtype=np.float64),
    D=np.array([-0.10364545969175058, -0.080709998495673896,
                0.0045663939845571044, 0.25845512445556379], dtype=np.float64).reshape(-1, 1),
    cv_roi=_JIQING_CV_ROI,
    H=None,  # BEV homography deferred to physical calibration
)

# Custom e-con camera — pinhole (Brown-Conrady k1,k2,p1,p2). TODO: fill in K, D,
# cv_roi, and H once the camera is calibrated and footage is recorded.
ECON = CameraConfig(
    name="econ",
    image_size=(1920, 1080),
    distortion_model="pinhole",
    K=None,
    D=None,
    cv_roi=None,
    H=None,
)
