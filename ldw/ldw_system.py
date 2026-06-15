"""Lane Departure Warning layer.

Consumes ANY detector that satisfies the LaneDetector contract (CV or ML),
projects its undistorted-pixel lane points into a single common bird's-eye view
via the calibrated homography H, applies temporal smoothing, and decides whether
the vehicle is departing its lane.

This is where state lives: the per-frame detectors are stateless, and all
temporal logic (rolling median of the lane centers) is concentrated here.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .detectors.base import LaneDetector, LaneList


@dataclass
class LDWResult:
    warning: bool
    center_bev: Optional[np.ndarray]      # smoothed lane midpoint in BEV, or None
    forward_ref_x: float                  # BEV x of the vehicle's forward axis
    lanes_px: LaneList                    # raw detector output (undistorted px)
    lanes_bev: List[np.ndarray]           # lanes projected into BEV


class LDWSystem:
    def __init__(
        self,
        detector: LaneDetector,
        H: np.ndarray,
        image_size,
        n_frames_median: int = 15,
        near_band_px: int = 100,
        center_threshold_px: float = 50.0,
    ):
        self.detector = detector
        self.H = np.asarray(H, dtype=np.float32)
        self.W, self.H_img = int(image_size[0]), int(image_size[1])
        self.n_frames_median = n_frames_median
        self.near_band_px = near_band_px
        self.center_threshold_px = center_threshold_px

        self._left_buf: List[np.ndarray] = []
        self._right_buf: List[np.ndarray] = []

        # Vehicle forward axis: the image's center column projected into BEV.
        self.forward_ref_x = self._forward_ref_x()

    def _to_bev(self, lane: np.ndarray) -> np.ndarray:
        if lane is None or len(lane) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = lane.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def _forward_ref_x(self) -> float:
        """BEV x-coordinate of the camera's forward (image-center) column."""
        col = np.array([[[self.W / 2.0, y]] for y in
                        np.linspace(self.H_img / 2.0, self.H_img - 1, 20)],
                       dtype=np.float32).reshape(-1, 1, 2)
        bev = cv2.perspectiveTransform(col, self.H).reshape(-1, 2)
        return float(np.mean(bev[:, 0]))

    def _near_center(self, lane_bev: np.ndarray) -> np.ndarray:
        """Mean position of the near-field part of a BEV lane (or NaNs)."""
        if len(lane_bev) == 0:
            return np.array([np.nan, np.nan], dtype=np.float32)
        near = lane_bev[lane_bev[:, 1] < (self.H_img / 2.0 + self.near_band_px)]
        if len(near) == 0:
            return np.array([np.nan, np.nan], dtype=np.float32)
        return near.mean(axis=0)

    def _smooth(self, buf: List[np.ndarray], value: np.ndarray) -> np.ndarray:
        buf.append(value)
        del buf[: -self.n_frames_median]
        return np.nanmedian(np.array(buf), axis=0)

    def process(self, frame_bgr: np.ndarray) -> LDWResult:
        lanes_px = self.detector.detect(frame_bgr)
        left_bev, right_bev = (self._to_bev(lanes_px[0]), self._to_bev(lanes_px[1]))

        left_c = self._smooth(self._left_buf, self._near_center(left_bev))
        right_c = self._smooth(self._right_buf, self._near_center(right_bev))

        center = None
        warning = False
        if not np.isnan(left_c).any() and not np.isnan(right_c).any():
            center = (left_c + right_c) / 2.0
            warning = abs(self.forward_ref_x - center[0]) > self.center_threshold_px

        return LDWResult(
            warning=warning,
            center_bev=center,
            forward_ref_x=self.forward_ref_x,
            lanes_px=lanes_px,
            lanes_bev=[left_bev, right_bev],
        )
