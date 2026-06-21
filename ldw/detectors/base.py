"""Shared lane-detector contract.

Both the CV and ML detectors implement the same interface so the LDW layer can
use either one interchangeably:

    detect(frame_bgr) -> [left_pts, right_pts]

where each element is an (N, 2) float32 array of (x, y) points in
**raw original-image pixels** — the same coordinate frame the image was passed
in (or an empty (0, 2) array when that lane is not found). Undistortion is NOT
applied to detector output: the LDW layer undistorts (with the camera's K/D)
and then projects into a common BEV via the homography H.
"""

from typing import List, Protocol

import numpy as np

# A lane is an (N, 2) array of (x, y) points; ego output is [left, right].
Lane = np.ndarray
LaneList = List[Lane]

EMPTY_LANE = np.zeros((0, 2), dtype=np.float32)


class LaneDetector(Protocol):
    """Structural interface for a lane detector."""

    def detect(self, frame_bgr: np.ndarray) -> LaneList:
        """Return [left_pts, right_pts] in undistorted original-image pixels."""
        ...
