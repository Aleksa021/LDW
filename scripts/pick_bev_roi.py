"""Interactive BEV-ROI picker for the Jiqing CV detector — fine keyboard control.

The 4 ego-lane corners (order BL, BR, TR, TL) start from the CURRENT cv_roi and
you nudge each one into place. The frame is undistorted with the EXACT same maps
the CVLaneDetector uses (fisheye, balance=0.0), so the points map 1:1 onto
cv_roi['src']. A live BEV preview updates as you move points — a good ROI makes
the ego-lane lines vertical and parallel there.

SELECT a point:   1=BL  2=BR  3=TR  4=TL   (or Tab to cycle)
MOVE it (fine, 1px):  arrow keys   or   h/j/k/l (left/down/up/right)
MOVE it (coarse,10px): H/J/K/L  (capitals)
STEP size:        [ decrease , ] increase   (shown on screen)
MOUSE:            left-click = jump the active point to the cursor
PRINT result:     p  or  ENTER   (prints "src": [...] to paste back)
RESET to current: r
QUIT:             q  or  Esc

Run (from anywhere, conda env with cv2):
    python /tmp/claude-1000/-media-aleksa21-NVme-Data-Projects-LDW/d23c0d66-7613-40f7-a184-a15ba483b91e/scratchpad/pick_bev_roi.py
"""
import argparse
import sys

import cv2
import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldw.cameras import JIQING
from ldw.detectors.cv_detector import CVLaneDetector

VIDEO = "/media/aleksa21/NVme Data/Datasets/Jiqing_Expressway/Videos/IMG_0310.MOV"
LABELS = ["BL", "BR", "TR", "TL"]
COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (255, 255, 0)]

# Starting trapezoid (BL, BR, TR, TL), fractions of (W, H). Edit / reset to this.
STARTING_SRC = [(0.0, 0.86), (0.9995, 0.8601), (0.5967, 0.6), (0.4429, 0.6)]

# key-code sets (cv2.waitKeyEx; arrow codes vary by backend, so accept several)
LEFT = {65361, 81, 2424832, ord("h")}
UP = {65362, 82, 2490368, ord("k")}
RIGHT = {65363, 83, 2555904, ord("l")}
DOWN = {65364, 84, 2621440, ord("j")}
LEFT_C, UP_C, RIGHT_C, DOWN_C = ord("H"), ord("K"), ord("L"), ord("J")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--frame", type=int, default=390, help="frame index (13s @ 29.97 = 390)")
    ap.add_argument("--scale", type=float, default=0.6, help="display scale factor")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {args.frame} from {args.video}")

    det = CVLaneDetector(image_size=JIQING.image_size, roi=JIQING.cv_roi,
                         calibration=JIQING.calibration,
                         distortion_model=JIQING.distortion_model)
    und = cv2.remap(frame, det.map1, det.map2, cv2.INTER_LINEAR)

    sc = args.scale
    disp_base = cv2.resize(und, (int(W * sc), int(H * sc)))
    dst = np.float32([(fx * W, fy * H) for fx, fy in JIQING.cv_roi["dst"]])

    # Vertical guide x-positions in the BEV = the dst left/right edges (target
    # lane positions). Align the warped lane lines onto these.
    guide_x = [int(dst[:, 0].min()), int(dst[:, 0].max())]

    def cur_pts():
        return [[fx * W, fy * H] for fx, fy in STARTING_SRC]

    pts = cur_pts()        # full-res [x, y]
    active = 0             # index of selected point
    step = 2               # px per fine move

    def clamp(p):
        p[0] = float(min(max(p[0], 0), W - 1))
        p[1] = float(min(max(p[1], 0), H - 1))

    def render():
        img = disp_base.copy()
        quad = np.float32(pts)
        cv2.polylines(img, [(quad * sc).astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            r = 8 if i == active else 5
            cv2.circle(img, (int(x * sc), int(y * sc)), r, COLORS[i], -1)
            if i == active:
                cv2.circle(img, (int(x * sc), int(y * sc)), r + 4, (255, 255, 255), 2)
            cv2.putText(img, LABELS[i], (int(x * sc) + 8, int(y * sc) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[i], 2)
        cv2.putText(img, f"active={LABELS[active]} ({int(pts[active][0])},{int(pts[active][1])})"
                         f"  step={step}px   1-4 select | arrows/hjkl move | p print | q quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow("pick BEV ROI (undistorted)", img)

        # live BEV preview with two vertical guide lines at the target lane x
        M = cv2.getPerspectiveTransform(np.float32(pts), dst)
        bev = cv2.warpPerspective(und, M, (W, H))
        for gx in guide_x:
            cv2.line(bev, (gx, 0), (gx, H - 1), (255, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("BEV preview (align lane lines onto magenta guides)",
                   cv2.resize(bev, (int(W * sc), int(H * sc))))

    def on_mouse(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN:
            pts[active] = [x / sc, y / sc]
            clamp(pts[active])
            render()

    cv2.namedWindow("pick BEV ROI (undistorted)")
    cv2.setMouseCallback("pick BEV ROI (undistorted)", on_mouse)
    render()

    def emit():
        fracs = [(round(x / W, 4), round(y / H, 4)) for x, y in pts]
        print("\n# --- cv_roi['src'] (BL, BR, TR, TL), fractions of (W,H) ---")
        print('"src":', [tuple(f) for f in fracs], ",")
        print("# raw px:", [(int(x), int(y)) for x, y in pts])

    while True:
        k = cv2.waitKeyEx(20)
        if k == -1:
            continue
        kk = k & 0xFF
        if kk in (ord("q"), 27):
            break
        elif kk in (ord("1"), ord("2"), ord("3"), ord("4")):
            active = kk - ord("1")
        elif kk == 9:  # Tab
            active = (active + 1) % 4
        elif kk == ord("["):
            step = max(1, step - 1)
        elif kk == ord("]"):
            step = min(50, step + 1)
        elif kk == ord("r"):
            pts = cur_pts()
        elif kk in (ord("p"), 13):
            emit()
        elif k in LEFT:
            pts[active][0] -= step
        elif k in RIGHT:
            pts[active][0] += step
        elif k in UP:
            pts[active][1] -= step
        elif k in DOWN:
            pts[active][1] += step
        elif kk == LEFT_C:
            pts[active][0] -= 10
        elif kk == RIGHT_C:
            pts[active][0] += 10
        elif kk == UP_C:
            pts[active][1] -= 10
        elif kk == DOWN_C:
            pts[active][1] += 10
        clamp(pts[active])
        render()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
