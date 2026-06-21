"""Lane-detection evaluation harness for the Jiqing Expressway dataset.

Scores either detector (CV or ML/UFLDv2) against the dataset's per-frame lane
ground truth using two complementary metrics:

  * CULane-style F1 at IoU 0.5 (lanes rendered as thick curves, matched by IoU).
  * TuSimple-style point accuracy + FP/FN rates (pred sampled at GT rows).

Scoring space (per the project's fairness choice):
  * ML  -> raw image pixels (GT used as-is); matches UFLDv2 published protocol.
  * CV  -> fisheye-undistorted pixels (frame AND GT undistorted with the dataset
           calibration), reflecting the CV detector's real pipeline.
Each detector is therefore scored in its own operating space.
"""
