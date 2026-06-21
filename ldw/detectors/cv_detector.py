"""CV lane detector: classic gradient/color + perspective + polynomial fit.

A self-contained, size-agnostic, stateless ego-lane detector conforming to the
LaneDetector contract: detect() returns the two ego-lane boundaries as points in
**original input-image pixels** (the same coordinate frame the image was passed
in).

Lens undistortion is handled internally: the frame is rectified for processing
(lanes are straight/parallel in a rectilinear view, which the perspective warp
and polynomial fit assume), then the detected lane points are re-distorted back
into the original image space before returning. Two distortion models are
supported via `distortion_model`:
  * "fisheye" - OpenCV fisheye model (4x1 D); used for the Jiqing Expressway cam.
  * "pinhole" - Brown-Conrady k1,k2,p1,p2 (and optional k3); for the e-con cam.
  * None      - no calibration; detect on the raw frame, output raw.

The algorithm originates from the cv_lane_detection submodule's reference
pipeline (gradient/HLS thresholding -> perspective warp -> sliding-window
polynomial fit), re-implemented here free of globals so the submodule can stay a
read-only reference. Everything past point extraction (curvature, off-center,
visualization) is intentionally omitted; it belongs to the LDW layer.

Pipeline (per frame, no state carried between calls):
    undistort -> downscale -> edge mask -> warp to internal BEV -> crop
    -> sliding-window search -> polynomial fit -> sample -> unwarp + upscale
    -> re-distort points to the original image space
"""

import cv2
import numpy as np

from .base import EMPTY_LANE, LaneList

__all__ = ["CVLaneDetector"]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# The perspective-transform trapezoid (`roi`) is camera-specific and supplied by
# the caller (see ldw/cameras.py); fractions of (W, H), order BL, BR, TR, TL.

# Intensity / gradient thresholds (resolution-independent).
DEFAULT_THRESHOLDS = {
    "s_thresh": (120, 255),
    "sx_thresh": (20, 100),
    "dir_thresh": (0.7, 1.3),
}

# Spatial constants as fractions of the WORKING width/height. These reproduce the
# original 1280x720 hardcoded values (working width was 320): 100/4=25 -> ~0.08*320.
DEFAULT_SPATIAL = {
    "margin_frac": 0.078,        # sliding-window half-width (frac of working W)
    "minpix_frac": 0.039,        # recenter threshold (frac of working W)
    "crop_left_frac": 0.117,     # left crop (frac of working W)
    "crop_right_frac": 0.0625,   # right crop (frac of working W)
    "parallel_std_frac": 0.066,  # max std of lane-width for a trusted fit
    "min_fit_pixels": 50,        # min inlier pixels (working res) to attempt a fit
}

NWINDOWS = 9


# ---------------------------------------------------------------------------
# Edge-detection helpers (free of globals)
# ---------------------------------------------------------------------------

def abs_sobel_thresh(img, orient="x", sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == "x":
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    else:
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled = np.uint8(255.0 * abs_sobel / np.max(abs_sobel)) if np.max(abs_sobel) > 0 else np.zeros_like(gray)
    binary = np.zeros_like(scaled)
    binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absdir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary = np.zeros_like(absdir)
    binary[(absdir >= thresh[0]) & (absdir <= thresh[1])] = 1
    return binary


def find_edges(img, thresholds=DEFAULT_THRESHOLDS):
    """BGR image -> weighted binary edge mask (same logic as lane.py:find_edges)."""
    s_thresh = thresholds["s_thresh"]
    sx_thresh = thresholds["sx_thresh"]
    dir_thresh = thresholds["dir_thresh"]

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(float)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    sxbinary = abs_sobel_thresh(img, orient="x", sobel_kernel=3, thresh=sx_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=3, thresh=dir_thresh)

    combined = np.zeros_like(s_channel)
    combined[((sxbinary == 1) & (dir_binary == 1)) | ((s_binary == 1) & (dir_binary == 1))] = 1

    c_bi = np.zeros_like(s_channel)
    c_bi[(sxbinary == 1) & (s_binary == 1)] = 2

    return combined + c_bi


# ---------------------------------------------------------------------------
# Sliding-window search (from lane.py:full_search, parametrized)
# ---------------------------------------------------------------------------

def _sliding_window_fit(binary_warped, margin, minpix, min_fit_pixels):
    """Returns (left_fit, right_fit); each is poly coeffs or None if not enough pixels."""
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    midpoint = np.int64(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int64(binary_warped.shape[0] / NWINDOWS)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []
    for window in range(NWINDOWS):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)
        if len(good_left) > minpix:
            leftx_current = np.int64(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            rightx_current = np.int64(np.mean(nonzerox[good_right]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    left_fit = None
    right_fit = None
    if len(left_lane_inds) >= min_fit_pixels:
        left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
    if len(right_lane_inds) >= min_fit_pixels:
        right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
    return left_fit, right_fit


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class CVLaneDetector:
    """Stateless CV ego-lane detector. Conforms to the LaneDetector contract.

    detect(image_bgr) -> [left_pts, right_pts]
        each is an (N, 2) float32 array of (x, y) in original-image pixels,
        or EMPTY_LANE (0, 2) when that lane is not reliably found.
    """

    def __init__(self, image_size, roi, calibration=None, distortion_model=None,
                 proc_downscale=4, edge_thresholds=DEFAULT_THRESHOLDS,
                 spatial=DEFAULT_SPATIAL, undistort_balance=0.0):
        if roi is None:
            raise ValueError("roi (camera perspective trapezoid) is required")
        self.W, self.H = int(image_size[0]), int(image_size[1])
        self.proc_downscale = proc_downscale
        self.edge_thresholds = edge_thresholds
        self.spatial = spatial

        # Working (processing) resolution.
        self.work_W = int(round(self.W / proc_downscale))
        self.work_H = int(round(self.H / proc_downscale))

        # Build perspective matrices at WORKING resolution from fractional ROI.
        src = np.float32([(fx * self.work_W, fy * self.work_H) for fx, fy in roi["src"]])
        dst = np.float32([(fx * self.work_W, fy * self.work_H) for fx, fy in roi["dst"]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

        # Calibration: precompute the rectify map (frame) + Knew (for re-distorting
        # detected points back to the original image). K/D are assumed to match
        # image_size. distortion_model selects the lens model.
        self.calibrated = calibration is not None
        if self.calibrated:
            if distortion_model not in ("fisheye", "pinhole"):
                raise ValueError(
                    "distortion_model must be 'fisheye' or 'pinhole' when calibration is given")
            self.dmodel = distortion_model
            self.K = np.asarray(calibration[0], dtype=np.float64).reshape(3, 3)
            self.D = np.asarray(calibration[1], dtype=np.float64).reshape(-1, 1)
            size = (self.W, self.H)
            if distortion_model == "fisheye":
                self.Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    self.K, self.D, size, np.eye(3), balance=undistort_balance)
                self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                    self.K, self.D, np.eye(3), self.Knew, size, cv2.CV_16SC2)
            else:  # pinhole (Brown-Conrady k1,k2,p1,p2[,k3])
                self.Knew, _ = cv2.getOptimalNewCameraMatrix(
                    self.K, self.D, size, undistort_balance, size)
                self.map1, self.map2 = cv2.initUndistortRectifyMap(
                    self.K, self.D, None, self.Knew, size, cv2.CV_16SC2)

        # Precompute spatial constants in working pixels.
        self.margin = max(1, int(round(spatial["margin_frac"] * self.work_W)))
        self.minpix = max(1, int(round(spatial["minpix_frac"] * self.work_W)))
        self.crop_l = int(round(spatial["crop_left_frac"] * self.work_W))
        self.crop_r = self.work_W - int(round(spatial["crop_right_frac"] * self.work_W))
        self.parallel_std = spatial["parallel_std_frac"] * self.work_W
        self.min_fit_pixels = spatial["min_fit_pixels"]

    def _redistort(self, pts):
        """Map points from the rectified (Knew) frame back to original pixels."""
        if pts is None or len(pts) == 0:
            return pts
        norm = (pts.astype(np.float64) - [self.Knew[0, 2], self.Knew[1, 2]]) \
            / [self.Knew[0, 0], self.Knew[1, 1]]
        if self.dmodel == "fisheye":
            d = cv2.fisheye.distortPoints(norm.reshape(1, -1, 2), self.K, self.D)
        else:  # pinhole: project normalized rays through K + D
            obj = np.hstack([norm, np.ones((len(norm), 1))]).reshape(-1, 1, 3)
            d, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), self.K, self.D)
        return d.reshape(-1, 2).astype(np.float32)

    def detect(self, image_bgr: np.ndarray) -> LaneList:
        # 1. Undistort the frame for processing (lanes become rectilinear).
        proc = cv2.remap(image_bgr, self.map1, self.map2, cv2.INTER_LINEAR) \
            if self.calibrated else image_bgr

        # 2. Downscale to working resolution.
        work = cv2.resize(proc, (self.work_W, self.work_H))

        # 3. Edge mask.
        mask = find_edges(work, self.edge_thresholds)

        # 4. Warp to internal BEV.
        warped = cv2.warpPerspective(mask, self.M, (self.work_W, self.work_H),
                                     flags=cv2.INTER_NEAREST)

        # 5. Crop side noise.
        binary_sub = np.zeros_like(warped)
        binary_sub[:, self.crop_l:self.crop_r] = warped[:, self.crop_l:self.crop_r]

        # 6. Sliding-window fit.
        left_fit, right_fit = _sliding_window_fit(
            binary_sub, self.margin, self.minpix, self.min_fit_pixels)

        # 7. Sample polynomials over the working BEV height.
        ploty = np.linspace(0, self.work_H - 1, self.work_H)
        left_fitx = self._eval(left_fit, ploty)
        right_fitx = self._eval(right_fit, ploty)

        # Parallelism gate: if both lanes found but inconsistent, distrust both.
        if left_fitx is not None and right_fitx is not None:
            if np.std(right_fitx - left_fitx) >= self.parallel_std:
                left_fitx = right_fitx = None

        # 8. Unwarp + upscale each lane to rectified full-res pixels.
        left_pts = self._to_rectified(left_fitx, ploty) if left_fitx is not None else EMPTY_LANE
        right_pts = self._to_rectified(right_fitx, ploty) if right_fitx is not None else EMPTY_LANE

        # 9. Re-distort back to the original input-image space.
        if self.calibrated:
            left_pts = self._redistort(left_pts)
            right_pts = self._redistort(right_pts)
        return [left_pts, right_pts]

    @staticmethod
    def _eval(fit, ploty):
        if fit is None:
            return None
        return fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]

    def _to_rectified(self, fitx, ploty):
        """BEV (working) polyline -> rectified full-res pixel points."""
        pts = np.stack([fitx, ploty], axis=1).reshape(-1, 1, 2).astype(np.float32)
        cam = cv2.perspectiveTransform(pts, self.M_inv).reshape(-1, 2)
        return cam * self.proc_downscale
