"""Lane Departure Warning layer.

Consumes ANY detector that satisfies the LaneDetector contract (CV or ML). The
detectors emit RAW original-image pixels, so this layer undistorts them (with the
camera's K/D) and then projects into a common bird's-eye view via the homography
H, and decides whether the vehicle is departing its lane.

Design choices (vs. the original midpoint heuristic):

* Self-calibrating metric. We do NOT require H to be metrically calibrated.
  Each frame the two ego lines give a known real-world separation (lane width,
  default 3.7 m), which yields a meters-per-pixel scale in BEV. So distances are
  reported in METERS without any external ground-plane calibration.

* Distance-to-each-line. At the near-car row we measure the lateral distance
  from the vehicle's forward axis to the left and right lines separately (the
  standard LDW signal), instead of collapsing both lanes to a midpoint.

* Per-line warning. A single detected line is enough to warn — important,
  because departure often makes the line you're crossing hard to detect.

* Local linear fit at a common reference row. The line position is read from a
  linear fit of the nearest-car BEV points evaluated at a shared y, so there is
  no image-space / BEV-space unit mix-up and no fixed pixel band.

* EMA smoothing of the metric outputs (low latency) instead of a long median.
"""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .cameras import CameraConfig
from .detectors.base import LaneDetector, LaneList

LANE_WIDTH_M = 3.7  # nominal lane width


@dataclass
class LDWResult:
    warning: bool
    dist_left_m: Optional[float]      # lateral distance vehicle axis -> left line (m)
    dist_right_m: Optional[float]     # lateral distance vehicle axis -> right line (m)
    offset_m: Optional[float]         # signed offset from lane center (+ = right of center)
    lanes_px: LaneList                # raw detector output (original-image px)
    lanes_bev: List[np.ndarray]       # lanes projected into BEV


class _EMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: Optional[float]) -> Optional[float]:
        if x is None:
            return self.value
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class LDWSystem:
    def __init__(
        self,
        detector: LaneDetector,
        camera: CameraConfig,
        lane_width_m: float = LANE_WIDTH_M,
        warn_margin_m: float = 1.0,      # warn when a line is closer than this
        near_pts: int = 10,              # # of nearest-car BEV points used for the line fit
        ema_alpha: float = 0.4,
    ):
        if camera.H is None:
            raise ValueError(
                f"camera '{camera.name}' has no homography H; BEV requires calibration")
        self.detector = detector
        self.H = np.asarray(camera.H, dtype=np.float32)
        self.W, self.H_img = int(camera.image_size[0]), int(camera.image_size[1])
        self._K = None if camera.K is None else np.asarray(camera.K, dtype=np.float64)
        self._D = None if camera.D is None else np.asarray(camera.D, dtype=np.float64)
        self._model = camera.distortion_model
        self.lane_width_m = lane_width_m
        self.warn_margin_m = warn_margin_m
        self.near_pts = near_pts

        self._scale = _EMA(ema_alpha)     # meters per BEV-pixel
        self._dl = _EMA(ema_alpha)
        self._dr = _EMA(ema_alpha)

        # Vehicle forward axis: image-center column, undistorted then projected to BEV.
        col = np.array([[self.W / 2.0, y] for y in np.linspace(0, self.H_img - 1, 30)],
                       dtype=np.float32)
        fwd = self._to_bev(col)
        self._fwd_fit = np.polyfit(fwd[:, 1], fwd[:, 0], 1)  # x = a*y + b

    # ---- geometry helpers ---------------------------------------------------
    def _undistort_points(self, lane: np.ndarray) -> np.ndarray:
        """Raw pixels -> undistorted pixels (same K frame). Identity if uncalibrated."""
        if self._K is None:
            return lane
        pts = lane.reshape(-1, 1, 2).astype(np.float64)
        if self._model == "fisheye":
            u = cv2.fisheye.undistortPoints(pts, self._K, self._D, P=self._K)
        else:
            u = cv2.undistortPoints(pts, self._K, self._D, P=self._K)
        return u.reshape(-1, 2).astype(np.float32)

    def _to_bev(self, lane: np.ndarray) -> np.ndarray:
        if lane is None or len(lane) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        und = self._undistort_points(lane).reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(und, self.H).reshape(-1, 2)

    def _line_fit(self, lane_bev: np.ndarray):
        """Linear fit x=f(y) from the nearest-car points; returns coeffs or None."""
        if len(lane_bev) < 2:
            return None
        near = lane_bev[np.argsort(lane_bev[:, 1])[-self.near_pts:]]  # largest y = closest
        return np.polyfit(near[:, 1], near[:, 0], 1)

    @staticmethod
    def _eval(fit, y):
        return float(np.polyval(fit, y)) if fit is not None else None

    # ---- main ---------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> LDWResult:
        lanes_px = self.detector.detect(frame_bgr)
        left_bev, right_bev = self._to_bev(lanes_px[0]), self._to_bev(lanes_px[1])

        left_fit, right_fit = self._line_fit(left_bev), self._line_fit(right_bev)

        # Reference row = nearest-car row both available lines can be evaluated at.
        maxes = [b[:, 1].max() for b, f in ((left_bev, left_fit), (right_bev, right_fit)) if f is not None]
        if not maxes:
            return LDWResult(False, self._dl.value, self._dr.value, None, lanes_px, [left_bev, right_bev])
        y_ref = float(min(maxes))

        left_x = self._eval(left_fit, y_ref)
        right_x = self._eval(right_fit, y_ref)
        fwd_x = self._eval(self._fwd_fit, y_ref)

        # Self-calibrate meters-per-pixel from the known lane width when both lines present.
        if left_x is not None and right_x is not None:
            width_px = right_x - left_x
            if width_px > 1e-3:
                self._scale.update(self.lane_width_m / width_px)
        scale = self._scale.value

        dist_left = dist_right = None
        if scale is not None:
            if left_x is not None:
                dist_left = self._dl.update((fwd_x - left_x) * scale)
            if right_x is not None:
                dist_right = self._dr.update((right_x - fwd_x) * scale)

        present = [d for d in (dist_left, dist_right) if d is not None]
        warning = bool(present) and min(present) < self.warn_margin_m
        offset = (dist_right - dist_left) / 2.0 if (dist_left is not None and dist_right is not None) else None

        return LDWResult(warning, dist_left, dist_right, offset, lanes_px, [left_bev, right_bev])
