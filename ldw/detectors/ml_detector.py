"""ML lane detector: Ultra-Fast-Lane-Detection-v2 (UFLDv2) on a TensorRT engine.

Refactored from the reference script trt_test_speed.py into a reusable detector
that conforms to the shared LaneDetector contract (see base.py): detect() returns
the two ego-lane boundaries as points in undistorted original-image pixels.

Two mapping bugs from the reference script are fixed here:

* Vertical (y) mapping. The reference spread the model's row classes uniformly
  over the whole cropped strip. In reality each class k sits at normalized input
  height v_k = (row_anchor[k] - (1 - crop_ratio)) / crop_ratio, with
  row_anchor = linspace(0.42, 1, num_cls_row) and crop_ratio = 0.6 (from the
  training config). Ignoring this placed every point too high (toward the
  horizon) by up to ~20 px. We now invert the true mapping.

* Horizontal (x) mapping. The reference took a global softmax over all grid
  cells. The trained/evaluated behaviour (deploy reference + demo.py) is a
  softmax over a local window around the argmax, plus a +0.5 cell offset. We use
  the local-window expectation here.
"""

import ctypes
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F

from .base import EMPTY_LANE, LaneList

# CUDA runtime (raw ctypes, no pycuda dependency).
_cudart = ctypes.CDLL("libcudart.so")
_cudart.cudaMalloc.restype = ctypes.c_int
_cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudart.cudaMemcpy.restype = ctypes.c_int
_cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

_CUDA_MEMCPY_D2D = 3
_CUDA_MEMCPY_D2H = 2


def _load_engine(engine_path: str):
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to load engine: {engine_path}")
    return engine


class _TRTModel:
    """Minimal TensorRT-10 wrapper (D2D input, D2H outputs) for the UFLDv2 engine."""

    def __init__(self, engine_path: str):
        self.engine = _load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)
        self.input_name = self.inputs[0]
        self.output_names = self.outputs

        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.input_nbytes = np.prod(self.input_shape) * np.dtype(self.input_dtype).itemsize

        self.bindings = []
        self.d_input = ctypes.c_void_p()
        self._malloc(self.d_input, self.input_nbytes, "input")
        self.bindings.append(self.d_input.value)

        self.output_dptrs = {}
        self.output_hbufs = {}
        for name in self.output_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
            self.output_hbufs[name] = np.empty(shape, dtype=dtype)
            dptr = ctypes.c_void_p()
            self._malloc(dptr, nbytes, name)
            self.output_dptrs[name] = (dptr, nbytes)
            self.bindings.append(dptr.value)

    @staticmethod
    def _malloc(ptr, nbytes, label):
        err = _cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(int(nbytes)))
        if err != 0:
            raise RuntimeError(f"cudaMalloc({label}) failed with code {err}")

    def infer_torch(self, inp: torch.Tensor) -> dict:
        """inp: 1xCxHxW CUDA tensor. Returns {name: host ndarray}."""
        inp = inp.contiguous()
        res = _cudart.cudaMemcpy(
            ctypes.c_void_p(self.d_input.value),
            ctypes.c_void_p(inp.data_ptr()),
            ctypes.c_size_t(int(self.input_nbytes)),
            _CUDA_MEMCPY_D2D,
        )
        if res != 0:
            raise RuntimeError(f"cudaMemcpy D2D failed with code {res}")

        self.context.set_tensor_address(self.input_name, self.d_input.value)
        for name, (dptr, _) in self.output_dptrs.items():
            self.context.set_tensor_address(name, dptr.value)
        self.context.execute_v2(self.bindings)

        outputs = {}
        for name, (dptr, nbytes) in self.output_dptrs.items():
            hbuf = self.output_hbufs[name]
            _cudart.cudaMemcpy(
                ctypes.c_void_p(hbuf.ctypes.data),
                ctypes.c_void_p(dptr.value),
                ctypes.c_size_t(int(nbytes)),
                _CUDA_MEMCPY_D2H,
            )
            outputs[name] = hbuf
        return outputs


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MLLaneDetector:
    """UFLDv2 ego-lane detector. Conforms to the LaneDetector contract."""

    def __init__(
        self,
        engine_path: str,
        image_size: Tuple[int, int] = (1920, 1080),
        calibration: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        crop_ratio: float = 0.6,
        row_anchor_start: float = 0.42,
        last_n_rows: int = 750,
        black_bar_ratio: float = 0.5,
        row_lane_idx: Tuple[int, int] = (1, 2),
        local_width: int = 1,
    ):
        self.model = _TRTModel(engine_path)
        self.W, self.H = int(image_size[0]), int(image_size[1])
        self.crop_ratio = crop_ratio
        self.row_anchor_start = row_anchor_start
        self.last_n_rows = last_n_rows
        self.black_bar_ratio = black_bar_ratio
        self.row_lane_idx = list(row_lane_idx)
        self.local_width = local_width

        self.K = self.D = None
        if calibration is not None:
            self.K = np.asarray(calibration[0], dtype=np.float32)
            self.D = np.asarray(calibration[1], dtype=np.float32)

        # Input geometry from the engine.
        self.train_height = int(self.model.input_shape[2])
        self.train_width = int(self.model.input_shape[3])

        loc_row_shape = self.model.engine.get_tensor_shape("loc_row")
        self.num_grid_row = int(loc_row_shape[1])
        self.num_cls_row = int(loc_row_shape[2])

        # Vertical anchors -> original-image y (the corrected mapping).
        row_anchor = np.linspace(self.row_anchor_start, 1.0, self.num_cls_row)
        v = (row_anchor - (1.0 - self.crop_ratio)) / self.crop_ratio  # normalized input height
        self.yi = (self.H - self.last_n_rows) + v * self.last_n_rows

    # ---- preprocessing (GPU) ------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        device = torch.device("cuda")
        frame = frame[-self.last_n_rows:, :, :]
        img = torch.from_numpy(frame).pin_memory().to(device, non_blocking=True, dtype=torch.float32)
        img = img / 255.0
        black_bar = int(self.train_width * self.black_bar_ratio)
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = F.interpolate(img, size=(self.train_height, self.train_width - black_bar),
                            mode="bilinear", align_corners=False)
        pad = black_bar // 2
        img = F.pad(img, (pad, pad, 0, 0), mode="constant", value=0.0)
        mean = torch.from_numpy(_MEAN).to(device).view(1, -1, 1, 1)
        std = torch.from_numpy(_STD).to(device).view(1, -1, 1, 1)
        return (img - mean) / std

    # ---- decoding -----------------------------------------------------------
    def _decode_x(self, loc_row: np.ndarray, k: int, lane: int) -> float:
        """Local-window softmax expectation over the grid dim -> original-image x."""
        col = loc_row[:, k, lane]                       # [num_grid_row]
        amax = int(np.argmax(col))
        lo = max(0, amax - self.local_width)
        hi = min(self.num_grid_row - 1, amax + self.local_width) + 1
        idx = np.arange(lo, hi)
        w = col[lo:hi]
        w = np.exp(w - w.max())
        w = w / w.sum()
        g = float((w * idx).sum()) + 0.5             # expected grid cell (+0.5 like the reference)

        # Remove black-bar padding, then scale to original width.
        u = g / (self.num_grid_row - 1)              # [0,1] across the padded input width
        ratio = self.black_bar_ratio
        u_img = (u - ratio / 2.0) / (1.0 - ratio)    # [0,1] across the image content
        return u_img * self.W

    def _pred2coords(self, pred: dict) -> LaneList:
        loc_row = pred["loc_row"][0]                 # [num_grid_row, num_cls_row, num_lane_row]
        exist_row = pred["exist_row"][0]             # [2, num_cls_row, num_lane_row]
        valid = exist_row.argmax(0).astype(bool)     # [num_cls_row, num_lane_row]

        lanes: List[np.ndarray] = []
        for lane in self.row_lane_idx:
            if valid[:, lane].sum() <= self.num_cls_row / 2:
                lanes.append(EMPTY_LANE)
                continue
            pts = [
                (self._decode_x(loc_row, k, lane), self.yi[k])
                for k in range(self.num_cls_row)
                if valid[k, lane]
            ]
            lanes.append(np.array(pts, dtype=np.float32) if pts else EMPTY_LANE)
        return lanes

    def _undistort(self, lane: np.ndarray) -> np.ndarray:
        if self.K is None or len(lane) == 0:
            return lane
        pts = lane.reshape(-1, 1, 2).astype(np.float32)
        return cv2.undistortPoints(pts, self.K, self.D, P=self.K).reshape(-1, 2)

    # ---- public API ---------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> LaneList:
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != (self.W, self.H):
            frame_bgr = cv2.resize(frame_bgr, (self.W, self.H))
        inp = self._preprocess(frame_bgr)
        outputs = self.model.infer_torch(inp)
        pred = {name: outputs[name] for name in self.model.output_names}
        coords = self._pred2coords(pred)
        return [self._undistort(lane) for lane in coords]
