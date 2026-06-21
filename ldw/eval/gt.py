"""Ground-truth parsing and ego-lane selection for the Jiqing dataset.

GT layout: one `<frame>.txt` per annotated frame in a per-video folder
(e.g. Lane_Parameters/0249/100.txt). Each line is:

    laneN: (x,y)(x,y)...

Points are pixel coords on the raw 1920x1080 frame and may run off-screen
(negative x). Lanes are ordered left->right; we keep only the two *ego*
boundaries (the pair straddling the vehicle / image center).
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

_PT = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_LANE = re.compile(r"lane\d+\s*:(.*)")


def parse_gt_file(path: str) -> List[np.ndarray]:
    """Return a list of lanes; each lane is an (N, 2) float32 array, y-sorted."""
    lanes: List[np.ndarray] = []
    with open(path, "r") as f:
        for line in f:
            m = _LANE.search(line)
            if not m:
                continue
            pts = [(float(x), float(y)) for x, y in _PT.findall(m.group(1))]
            if len(pts) < 2:
                continue
            arr = np.array(pts, dtype=np.float32)
            arr = arr[np.argsort(arr[:, 1])]  # sort by y (top -> bottom)
            lanes.append(arr)
    return lanes


def _x_at(lane: np.ndarray, y: float) -> Optional[float]:
    """Interpolate lane x at row y, or None if y is outside the lane's range."""
    ys, xs = lane[:, 1], lane[:, 0]
    if y < ys.min() or y > ys.max():
        return None
    return float(np.interp(y, ys, xs))


def select_ego_pair(lanes: List[np.ndarray], image_width: int,
                    image_height: int, center_x: Optional[float] = None
                    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Pick the (left, right) ego boundaries: nearest lanes either side of center.

    Evaluated at a reference row near the image bottom where lanes are on-screen.
    Returns (left_lane, right_lane); either may be None if absent on that side.
    """
    if not lanes:
        return None, None
    cx = image_width / 2.0 if center_x is None else center_x
    y_ref = 0.85 * image_height

    left = right = None
    left_x = -np.inf
    right_x = np.inf
    for lane in lanes:
        x = _x_at(lane, y_ref)
        if x is None:
            # Fall back to the lane's lowest (largest-y) point.
            x = float(lane[-1, 0])
        if x <= cx and x > left_x:
            left_x, left = x, lane
        elif x > cx and x < right_x:
            right_x, right = x, lane
    return left, right


def load_video_gt(gt_dir: str) -> Dict[int, str]:
    """Map annotated frame index -> GT file path for one video folder."""
    out: Dict[int, str] = {}
    for name in os.listdir(gt_dir):
        if name.endswith(".txt"):
            try:
                out[int(os.path.splitext(name)[0])] = os.path.join(gt_dir, name)
            except ValueError:
                continue
    return out
