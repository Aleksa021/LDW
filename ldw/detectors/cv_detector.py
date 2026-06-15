"""CV lane detector adapter.

Re-exports the size-agnostic CVLaneDetector that lives in the cv_lane_detection
submodule so it can be used through the shared LaneDetector contract. The
submodule isn't an installable package, so we add it to sys.path on import.

CVLaneDetector already returns [left_pts, right_pts] in undistorted
original-image pixels, so it satisfies the contract directly.
"""

import os
import sys

_CV_SUBMODULE = os.path.join(
    os.path.dirname(__file__), "..", "..", "submodules", "cv_lane_detection"
)
_CV_SUBMODULE = os.path.abspath(_CV_SUBMODULE)
if _CV_SUBMODULE not in sys.path:
    sys.path.insert(0, _CV_SUBMODULE)

from cv_lane_module import CVLaneDetector  # noqa: E402

__all__ = ["CVLaneDetector"]
