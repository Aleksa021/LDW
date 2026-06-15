"""Run either lane detector through the shared LDWSystem.

    python -m ldw.demo --detector ml        # UFLDv2 TensorRT on centar_grada_kraci.mp4
    python -m ldw.demo --detector cv        # CV pipeline on project_video.mp4
    python -m ldw.demo --detector cv --frames 30 --no-show   # headless sanity run

Both detectors return undistorted original-pixel lane points; LDWSystem projects
them into each camera's BEV via that camera's homography H, smooths, and warns.
The two configs use different cameras/videos — to truly compare CV vs ML, feed
the SAME footage + H to both.
"""

import argparse
import os

import cv2
import numpy as np

from ldw.ldw_system import LDWSystem

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML_DIR = os.path.join(ROOT, "submodules", "ml_lane_detection")
CV_DIR = os.path.join(ROOT, "submodules", "cv_lane_detection")

# ---- ML camera (from the reference trt_test_speed.py) ----------------------
ML_K = np.array([[1595.670753182030012, 0, 954.9628573763222903],
                 [0, 1602.230802012541062, 538.4237247698257534],
                 [0, 0, 1]], dtype=np.float32)
ML_D = np.array([-0.3623766942436448812, 0.1749287771660668622,
                 -0.001563359095306956397, 0.0006625874294289102549,
                 -0.05333212038737372013], dtype=np.float32)
ML_H = np.array([[-0.9699118924366116890, -6.692142227856339609, 2394.451484153518322],
                 [-0.1127449670934883574, -1.488161138959835261, -1236.693230365772251],
                 [2.404776845372572024e-05, -0.006943593727037860111, 1.300326227112413413]],
                dtype=np.float32)


def build_ml():
    from ldw.detectors.ml_detector import MLLaneDetector
    size = (1920, 1080)
    det = MLLaneDetector(
        engine_path=os.path.join(ML_DIR, "resources", "culane_res34.engine"),
        image_size=size,
        calibration=(ML_K, ML_D),
    )
    sys_ = LDWSystem(det, ML_H, image_size=size)
    video = os.path.join(ML_DIR, "centar_grada_kraci.mp4")
    return sys_, video, (ML_K, ML_D)


def build_cv():
    from ldw.detectors.cv_detector import CVLaneDetector
    import sys
    if CV_DIR not in sys.path:
        sys.path.insert(0, CV_DIR)
    from calibration import load_calibration

    size = (1280, 720)
    mtx, dist = load_calibration(os.path.join(CV_DIR, "camera_cal", "calibration_pickle.p"))
    det = CVLaneDetector(image_size=size, calibration=(mtx, dist))

    # Homography for the CV camera: the original lane.py trapezoid at full res.
    src = np.float32([[194, 719], [1117, 719], [705, 461], [575, 461]])
    dst = np.float32([[290, 719], [990, 719], [990, 0], [290, 0]])
    H_cv = cv2.getPerspectiveTransform(src, dst)

    sys_ = LDWSystem(det, H_cv, image_size=size)
    video = os.path.join(CV_DIR, "examples", "project_video.mp4")
    return sys_, video, (mtx, dist)


LANE_COLORS = [(0, 0, 255), (255, 0, 0)]  # left=red, right=blue


def draw(frame, result):
    for pts, color in zip(result.lanes_px, LANE_COLORS):
        if len(pts) == 0:
            continue
        poly = pts.astype(np.int32)
        cv2.polylines(frame, [poly], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)
        for x, y in poly:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    def fmt(v):
        return f"{v:+.2f}m" if v is not None else "--"
    cv2.putText(frame, f"L:{fmt(result.dist_left_m)}  R:{fmt(result.dist_right_m)}  off:{fmt(result.offset_m)}",
                (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    if result.warning:
        cv2.putText(frame, "WARNING: Lane Departure!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    return frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", choices=["cv", "ml"], required=True)
    p.add_argument("--video", default=None, help="override input video path")
    p.add_argument("--frames", type=int, default=0, help="process at most N frames (0 = all)")
    p.add_argument("--no-show", action="store_true", help="headless; print stats only")
    args = p.parse_args()

    system, video, calib = build_ml() if args.detector == "ml" else build_cv()
    if args.video:
        video = args.video
    mtx, dist = calib

    cap = cv2.VideoCapture(video)
    n, warnings, found = 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = system.process(frame)
        n += 1
        warnings += int(result.warning)
        found += int(len(result.lanes_px[0]) > 0 and len(result.lanes_px[1]) > 0)

        if not args.no_show:
            view = cv2.undistort(cv2.resize(frame, (system.W, system.H_img)), mtx, dist, None, mtx)
            view = draw(view, result)
            cv2.imshow(f"LDW [{args.detector}]", cv2.resize(view, (1280, 720)))
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if args.frames and n >= args.frames:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[{args.detector}] frames={n} both-lanes-found={found} warnings={warnings}")


if __name__ == "__main__":
    main()
