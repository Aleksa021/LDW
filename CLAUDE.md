# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LDW (Lane Departure Warning) is a real-time lane detection and departure warning system with two submodule implementations:

- `submodules/ml_lane_detection` — Ultra-Fast Lane Detection v2 (UFLDv2), a deep learning approach using ResNet backbones trained on CULane/Tusimple/CurveLanes datasets.
- `submodules/cv_lane_detection` — Traditional computer vision approach using gradient/color thresholding, perspective transforms, and polynomial fitting.

The root-level `trt_test_speed.py` is the primary production inference script, combining the UFLDv2 model (deployed as a TensorRT engine) with custom lane departure logic.

## Commands

**Run TRT inference (main script):**
```bash
python trt_test_speed.py
```
Requires `submodules/ml_lane_detection/resources/culane_res34.engine` and `submodules/ml_lane_detection/centar_grada_kraci.mp4` to exist.

**Run CV-based lane detection:**
```bash
cd submodules/cv_lane_detection && python main.py
```

**Export PyTorch model to ONNX:**
```bash
cd submodules/ml_lane_detection
python deploy/pt2onnx.py --config_path configs/culane_res34.py --model_path weights/culane_res34.pth
```

**Convert ONNX to TRT engine** (run from `ml_lane_detection` dir):
```bash
trtexec --onnx=weights/culane_res34.onnx --saveEngine=resources/culane_res34.engine --fp16
```

**Build custom interpolation C extension** (required for training):
```bash
cd submodules/ml_lane_detection/my_interp && sh build.sh
```

**Install ML submodule dependencies** (Python 3.7, CUDA 11.7):
```bash
conda create -n lane-det python=3.7 -y && conda activate lane-det
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia/label/cuda-11.7.1
pip install -r submodules/ml_lane_detection/requirements.txt
```

## Architecture of `trt_test_speed.py`

### Inference pipeline

```
Video frame
  → preprocess_torch()        # GPU: crop last_n_rows, resize, pad black bars, normalize
  → TRTModel.infer_torch()    # D2D copy into TRT input buffer, execute_v2, D2H copy outputs
  → pred2coords()             # softmax over loc_row logits → weighted sum → pixel x-coords
  → cv2.undistortPoints()     # remove lens distortion using K, D
  → cv2.perspectiveTransform(H)  # project into Bird's Eye View (BEV) space
  → temporal median filter    # 15-frame rolling median on left/right lane centers in BEV
  → departure check           # compare BEV lane midpoint to image center; warn if > 50px
```

### `Config` dataclass

Holds all calibration and inference constants. Key fields:
- `K`, `D` — camera intrinsic matrix and distortion coefficients (hardcoded for the specific camera)
- `H`, `H_inv` — homography matrix for BEV projection (and its inverse for back-projecting to image coords)
- `train_height`, `train_width` — auto-populated from the TRT engine's input tensor shape
- `num_grid_row`, `num_cls_row`, etc. — auto-populated from engine output tensor shapes via `set_output_shapes()`
- `last_n_rows` — only the bottom 750 rows of the 1080p frame are fed to the model
- `black_bar_ratio` — 50% of the network input width is padded with black bars on each side; only the center strip contains the resized crop. This corrects for the wide aspect of the original frame.

### `TRTModel` class

Uses the **TensorRT 10 API** (`get_tensor_name`, `set_tensor_address`, `execute_v2`) — not the deprecated binding-based API used in `deploy/trt_infer.py`. GPU memory is managed via raw `ctypes` calls to `libcudart.so` (no pycuda dependency). Two inference paths:
- `infer(inp)` — host numpy array input (H2D copy)
- `infer_torch(inp)` — CUDA tensor input (D2D copy, avoids PCIe round-trip)

### Model outputs

The UFLDv2 engine produces four tensors:
- `loc_row` — `[1, num_grid_row, num_cls_row, num_lane_row]` — horizontal position logits for row-anchored lanes
- `loc_col` — `[1, num_grid_col, num_cls_col, num_lane_col]` — vertical position logits for col-anchored lanes
- `exist_row` / `exist_col` — `[1, 2, num_cls, num_lane]` — binary lane existence logits

`row_lane_idx = [1, 2]` selects the two ego lanes (left and right of ego); `col_lane_idx = [0, 3]` selects outer lanes. Only row lanes are used for departure detection in the current implementation.

## CV Submodule (`cv_lane_detection`)

Pipeline in `lane.py → process_frame()`:
1. Undistort with stored calibration (`camera_cal/calibration_pickle.p`)
2. Edge detection: HLS S-channel thresholding + Sobel gradient → combined binary mask
3. Perspective warp to bird's eye view using hardcoded `src`/`dst` point pairs
4. Lane fitting: `detector()` (full sliding-window search) or `tracker()` (search around prior polynomial) depending on `Lane.detected` state
5. Polynomial → curvature + off-center distance computation
6. Departure warning if off-center > 0.6 m

## Submodule Management

```bash
git submodule update --init --recursive   # initialize after clone
git submodule update --remote             # pull latest from upstream
```

`cv_lane_detection` is currently untracked (shows as `?` in git status). `ml_lane_detection` has local modifications relative to its upstream commit.
