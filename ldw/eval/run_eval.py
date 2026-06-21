"""Evaluate a lane detector against the Jiqing Expressway dataset.

Examples
--------
# Visual sanity check first (dump overlays, no metrics):
python -m ldw.eval.run_eval --detector ml-culane --videos 0249 \
       --overlay-dir /tmp/ov --overlay-only --max-frames 5

# Score a subset of clips, every 5th frame:
python -m ldw.eval.run_eval --detector cv --videos 0249 0250 0251 \
       --stride 5 --out results_cv.json

Detectors: cv | ml-culane | ml-tusimple.
Scoring space: ml-* in raw pixels; cv in fisheye-undistorted pixels (frame and
GT undistorted with the dataset calibration).
"""

import argparse
import json
import os
from typing import List, Optional

import cv2
import numpy as np

from ldw.eval.calib import FisheyeRectifier, load_parameters
from ldw.eval.gt import load_video_gt, parse_gt_file, select_ego_pair
from ldw.eval.metrics import CULaneAccumulator, TuSimpleAccumulator, clip_to_yrange

DATASET = "/media/aleksa21/NVme Data/Datasets/Jiqing_Expressway"
VIDEO_DIR = os.path.join(DATASET, "Videos")
GT_ROOT = os.path.join(DATASET, "Lane_Parameters")
PARAMS = os.path.join(DATASET, "parameters.yml")

_RES = os.path.join(os.path.dirname(__file__), "..", "..",
                    "submodules", "ml_lane_detection", "resources")
ENGINES = {
    "ml-culane": os.path.abspath(os.path.join(_RES, "culane_res34.engine")),
    "ml-tusimple": os.path.abspath(os.path.join(_RES, "tusimple_res34.engine")),
}


def video_path(vid: str) -> str:
    return os.path.join(VIDEO_DIR, f"IMG_{vid}.MOV")


# CV perspective trapezoid tuned for the Jiqing camera in fisheye-undistorted
# space (fractions of W,H; order BL, BR, TR, TL). Derived from the undistorted
# ego-lane geometry; the detector's own DEFAULT_ROI is left untouched.
JIQING_CV_ROI = {
    "src": [(0.22, 0.86), (0.78, 0.86), (0.575, 0.60), (0.445, 0.60)],
    "dst": [(0.2266, 1.0), (0.7734, 1.0), (0.7734, 0.0), (0.2266, 0.0)],
}


def build_detector(kind: str, size):
    """Return (detector, undistort_for_cv: bool). Detectors run calibration=None."""
    W, H = size
    if kind == "cv":
        from ldw.detectors.cv_detector import CVLaneDetector
        return CVLaneDetector(image_size=(W, H), calibration=None, roi=JIQING_CV_ROI), True
    from ldw.detectors.ml_detector import MLLaneDetector
    if kind == "ml-culane":
        det = MLLaneDetector(
            ENGINES[kind], image_size=(W, H), calibration=None,
            crop_ratio=0.6, row_anchor_start=0.42, row_anchor_end=1.0,
            last_n_rows=int(0.6 * H), black_bar_ratio=0.0, row_lane_idx=(1, 2))
    elif kind == "ml-tusimple":
        det = MLLaneDetector(
            ENGINES[kind], image_size=(W, H), calibration=None,
            crop_ratio=0.8, row_anchor_start=160 / 720, row_anchor_end=710 / 720,
            last_n_rows=int(0.8 * H), black_bar_ratio=0.0, row_lane_idx=(1, 2))
    else:
        raise ValueError(f"unknown detector: {kind}")
    return det, False


def _nonempty(lane: np.ndarray) -> Optional[np.ndarray]:
    return lane if (lane is not None and len(lane) >= 2) else None


def _draw(img, lanes, colors, thick=4):
    for lane, c in zip(lanes, colors):
        if lane is not None and len(lane) >= 2:
            cv2.polylines(img, [lane.astype(np.int32)], False, c, thick, cv2.LINE_AA)
    return img


def evaluate(kind: str, videos: List[str], stride: int, max_frames: int,
             frame_offset: int, overlay_dir: Optional[str], overlay_only: bool,
             line_width: int, pix_thresh: float) -> dict:
    cap0 = cv2.VideoCapture(video_path(videos[0]))
    W = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    detector, do_undistort = build_detector(kind, (W, H))
    rect = FisheyeRectifier(*load_parameters(PARAMS), image_size=(W, H)) if do_undistort else None

    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)

    per_video = {}
    overall_cu = CULaneAccumulator(image_shape=(H, W), line_width=line_width)
    overall_ts = TuSimpleAccumulator(pix_thresh=pix_thresh)

    for vid in videos:
        gt_map = load_video_gt(os.path.join(GT_ROOT, vid))
        frames = sorted(gt_map)[::stride]
        if max_frames:
            frames = frames[:max_frames]

        cap = cv2.VideoCapture(video_path(vid))
        cu = CULaneAccumulator(image_shape=(H, W), line_width=line_width)
        ts = TuSimpleAccumulator(pix_thresh=pix_thresh)
        n_done = 0

        for fidx in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx + frame_offset)
            ok, frame = cap.read()
            if not ok:
                continue

            gt_lanes = parse_gt_file(gt_map[fidx])
            if do_undistort:
                frame = rect.undistort_image(frame)
                gt_lanes = [rect.undistort_points(l) for l in gt_lanes]

            gt_left, gt_right = select_ego_pair(gt_lanes, W, H)
            pred = detector.detect(frame)
            pred_left = _nonempty(pred[0]) if len(pred) > 0 else None
            pred_right = _nonempty(pred[1]) if len(pred) > 1 else None

            if not overlay_only:
                # CULane IoU: clip predictions to the GT's annotated y-band per side.
                cu_pred = [clip_to_yrange(pred_left, gt_left),
                           clip_to_yrange(pred_right, gt_right)]
                cu.add_frame(cu_pred, [gt_left, gt_right])
                overall_cu.add_frame(cu_pred, [gt_left, gt_right])
                for p, g in ((pred_left, gt_left), (pred_right, gt_right)):
                    ts.add_pair(p, g)
                    overall_ts.add_pair(p, g)

            if overlay_dir and n_done < 12:
                vis = frame.copy()
                _draw(vis, [gt_left, gt_right], [(0, 255, 255), (0, 200, 200)], 8)
                _draw(vis, [pred_left, pred_right], [(0, 0, 255), (255, 0, 0)], 3)
                cv2.imwrite(os.path.join(overlay_dir, f"{kind}_{vid}_{fidx}.png"), vis)
            n_done += 1

        cap.release()
        per_video[vid] = {"frames": n_done, "culane": cu.summary(), "tusimple": ts.summary()}
        print(f"[{kind}] {vid}: {n_done} frames | F1={cu.f1:.3f} "
              f"P={cu.precision:.3f} R={cu.recall:.3f} | "
              f"acc={ts.accuracy:.3f} FP={ts.fp_rate:.3f} FN={ts.fn_rate:.3f}")

    return {
        "detector": kind,
        "videos": videos,
        "stride": stride,
        "scoring_space": "fisheye_undistorted" if do_undistort else "raw",
        "line_width": line_width,
        "pix_thresh": pix_thresh,
        "overall": {"culane": overall_cu.summary(), "tusimple": overall_ts.summary()},
        "per_video": per_video,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector", required=True, choices=["cv", "ml-culane", "ml-tusimple"])
    p.add_argument("--videos", nargs="+", required=True, help="video ids, e.g. 0249 0250")
    p.add_argument("--stride", type=int, default=1, help="evaluate every Nth annotated frame")
    p.add_argument("--max-frames", type=int, default=0, help="cap frames per video (0=all)")
    p.add_argument("--frame-offset", type=int, default=0, help="video_frame = gt_frame + offset")
    p.add_argument("--line-width", type=int, default=30, help="CULane curve width (px)")
    p.add_argument("--pix-thresh", type=float, default=20.0, help="TuSimple hit threshold (px)")
    p.add_argument("--overlay-dir", default=None, help="dump verification overlays here")
    p.add_argument("--overlay-only", action="store_true", help="dump overlays, skip metrics")
    p.add_argument("--out", default=None, help="write JSON results here")
    args = p.parse_args()

    res = evaluate(args.detector, args.videos, args.stride, args.max_frames,
                   args.frame_offset, args.overlay_dir, args.overlay_only,
                   args.line_width, args.pix_thresh)

    if not args.overlay_only:
        ov = res["overall"]
        print(f"\n== {args.detector} OVERALL ({res['scoring_space']}) ==")
        print(f"CULane  F1={ov['culane']['f1']:.4f}  P={ov['culane']['precision']:.4f}  "
              f"R={ov['culane']['recall']:.4f}  (TP={ov['culane']['tp']} "
              f"FP={ov['culane']['fp']} FN={ov['culane']['fn']})")
        print(f"TuSimple acc={ov['tusimple']['accuracy']:.4f}  "
              f"pt_acc={ov['tusimple']['point_accuracy']:.4f}  "
              f"FP={ov['tusimple']['fp_rate']:.4f}  FN={ov['tusimple']['fn_rate']:.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
