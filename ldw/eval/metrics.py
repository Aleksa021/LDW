"""Lane-detection metrics: CULane-style F1@IoU and TuSimple-style accuracy.

Both operate on lanes represented as (N, 2) float arrays of (x, y) pixel points.
A "side-aligned" view is used throughout: predictions and GT are each reduced to
an optional (left, right) ego pair, so matching is by side rather than global.

CULane F1:   render each lane as a thick curve, match pred<->GT by mask IoU,
             count a hit at IoU > 0.5, accumulate TP/FP/FN -> P, R, F1.
TuSimple:    sample the predicted curve at the GT's row positions; a point is
             correct within `pix_thresh` px. Per-lane accuracy >= 0.85 => matched;
             report mean point accuracy plus FP/FN lane rates.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

Lane = Optional[np.ndarray]


def clip_to_yrange(lane: Lane, ref: Lane) -> Lane:
    """Restrict `lane` to the y-range of `ref` (with interpolated endpoints).

    The dataset annotates lanes over a shorter y-span than the detectors predict.
    Comparing full-length predictions against short GT by thick-curve IoU is
    unfair (union is dominated by the un-annotated prediction tail). Clipping the
    prediction to the GT's annotated band makes IoU measure lateral agreement
    where ground truth actually exists.
    """
    if lane is None or len(lane) < 2 or ref is None or len(ref) < 2:
        return lane
    y0, y1 = float(ref[:, 1].min()), float(ref[:, 1].max())
    ly = lane[:, 1]
    order = np.argsort(ly)
    sl = lane[order]
    sly, slx = sl[:, 1], sl[:, 0]
    inside = sl[(sly >= y0) & (sly <= y1)]
    pts = [inside]
    # Add interpolated boundary points so the clipped curve reaches the band edges.
    if sly.min() <= y0 <= sly.max():
        pts.append(np.array([[np.interp(y0, sly, slx), y0]], dtype=lane.dtype))
    if sly.min() <= y1 <= sly.max():
        pts.append(np.array([[np.interp(y1, sly, slx), y1]], dtype=lane.dtype))
    out = np.vstack([p for p in pts if len(p)])
    return out[np.argsort(out[:, 1])] if len(out) >= 2 else None


# ---------------------------------------------------------------------------
# CULane-style F1 at IoU 0.5
# ---------------------------------------------------------------------------

def _lane_mask(lane: Lane, shape: Tuple[int, int], width: int) -> Optional[np.ndarray]:
    if lane is None or len(lane) < 2:
        return None
    m = np.zeros(shape, dtype=np.uint8)
    pts = lane.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(m, [pts], False, 1, thickness=width, lineType=cv2.LINE_8)
    return m


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 0.0


@dataclass
class CULaneAccumulator:
    """Aggregates TP/FP/FN across frames for F1 at a fixed IoU threshold."""
    image_shape: Tuple[int, int]          # (H, W)
    line_width: int = 30
    iou_thresh: float = 0.5
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add_frame(self, preds: List[Lane], gts: List[Lane]) -> Tuple[int, int, int]:
        pmasks = [m for m in (_lane_mask(p, self.image_shape, self.line_width) for p in preds) if m is not None]
        gmasks = [m for m in (_lane_mask(g, self.image_shape, self.line_width) for g in gts) if m is not None]

        # Greedy IoU matching (>= handles the <=2-lane ego case exactly).
        scored = sorted(
            ((_iou(pmasks[i], gmasks[j]), i, j)
             for i in range(len(pmasks)) for j in range(len(gmasks))),
            reverse=True,
        )
        used_p, used_g = set(), set()
        tp = 0
        for v, i, j in scored:
            if v <= self.iou_thresh:
                break
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            tp += 1
        fp = len(pmasks) - tp
        fn = len(gmasks) - tp
        self.tp += tp
        self.fp += fp
        self.fn += fn
        return tp, fp, fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def summary(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": self.precision, "recall": self.recall, "f1": self.f1}


# ---------------------------------------------------------------------------
# TuSimple-style point accuracy + FP/FN
# ---------------------------------------------------------------------------

def _sample_x(lane: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Interpolate lane x at rows `ys`; NaN where ys is outside the lane range."""
    ly, lx = lane[:, 1], lane[:, 0]
    order = np.argsort(ly)
    ly, lx = ly[order], lx[order]
    x = np.interp(ys, ly, lx)
    x[(ys < ly.min()) | (ys > ly.max())] = np.nan
    return x


@dataclass
class TuSimpleAccumulator:
    """Aggregates point accuracy and lane FP/FN across frames (TuSimple protocol)."""
    pix_thresh: float = 20.0
    accept: float = 0.85          # per-lane accuracy to count as matched
    correct_pts: int = 0
    total_pts: int = 0
    fp_lanes: int = 0             # predicted lanes that don't match a GT lane
    fn_lanes: int = 0             # GT lanes with no matching prediction
    n_pred: int = 0
    n_gt: int = 0
    acc_sum: float = 0.0          # sum of per-GT-lane accuracies (for mean accuracy)
    n_acc: int = 0

    def add_pair(self, pred: Lane, gt: Lane) -> None:
        """Score one side (left or right). Either may be None/empty."""
        has_pred = pred is not None and len(pred) >= 2
        has_gt = gt is not None and len(gt) >= 2
        if has_pred:
            self.n_pred += 1
        if has_gt:
            self.n_gt += 1

        if not has_gt:
            if has_pred:
                self.fp_lanes += 1     # predicted a lane where there is none
            return
        if not has_pred:
            self.fn_lanes += 1         # missed an existing lane
            self.n_acc += 1            # accuracy 0 for this GT lane
            return

        ys = gt[:, 1]
        xs_gt = gt[:, 0]
        xs_pred = _sample_x(pred, ys)
        valid = ~np.isnan(xs_pred)
        correct = int(np.count_nonzero(valid & (np.abs(xs_pred - xs_gt) <= self.pix_thresh)))
        total = len(ys)
        self.correct_pts += correct
        self.total_pts += total
        acc = correct / total if total else 0.0
        self.acc_sum += acc
        self.n_acc += 1
        if acc < self.accept:
            self.fp_lanes += 1
            self.fn_lanes += 1

    @property
    def accuracy(self) -> float:
        """Mean per-lane point accuracy (TuSimple's headline 'Accuracy')."""
        return self.acc_sum / self.n_acc if self.n_acc else 0.0

    @property
    def point_accuracy(self) -> float:
        """Pooled correct/total points (localization precision)."""
        return self.correct_pts / self.total_pts if self.total_pts else 0.0

    @property
    def fp_rate(self) -> float:
        return self.fp_lanes / self.n_pred if self.n_pred else 0.0

    @property
    def fn_rate(self) -> float:
        return self.fn_lanes / self.n_gt if self.n_gt else 0.0

    def summary(self) -> dict:
        return {"accuracy": self.accuracy, "point_accuracy": self.point_accuracy,
                "fp_rate": self.fp_rate, "fn_rate": self.fn_rate,
                "n_pred": self.n_pred, "n_gt": self.n_gt}
