"""ML lane detector: Ultra-Fast-Lane-Detection-v2 (UFLDv2) on a TensorRT engine.

Conforms to the LaneDetector contract: detect() returns the two ego-lane
boundaries as points in RAW original-image pixels (no undistortion — that is the
LDW layer's job).

Preprocessing replicates UFLDv2's: resize the full frame width to the engine
input and feed the bottom `crop_ratio` of the image. Decoding then maps each row
class to its original-image row (y = row_anchor * H) and decodes x as a
softmax-weighted mean grid index over a local window around the argmax. The two
supported engines differ only in `crop_ratio` and `row_anchor` (see
ENGINE_PRESETS), selected by the `dataset` argument.
"""

import ctypes

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

# The two engines this project uses. row_anchor = (start, end) as fractions of
# image height (UFLDv2's training anchors, utils/common.py); crop_ratio is the
# bottom fraction of the frame fed to the network.
ENGINE_PRESETS = {
    "culane":   {"crop_ratio": 0.6, "row_anchor": (0.42, 1.0)},
    "tusimple": {"crop_ratio": 0.8, "row_anchor": (160 / 720, 710 / 720)},
}
ROW_LANE_IDX = (1, 2)  # the two ego lanes (left, right) in UFLDv2's lane ordering


class MLLaneDetector:
    """UFLDv2 ego-lane detector. Conforms to the LaneDetector contract.

    Returns the two ego-lane boundaries as raw original-image pixels. `dataset`
    selects the engine decode preset ("culane" or "tusimple").
    """

    def __init__(self, engine_path, dataset, image_size=(1920, 1080), local_width=1):
        if dataset not in ENGINE_PRESETS:
            raise ValueError(f"dataset must be one of {list(ENGINE_PRESETS)}")
        preset = ENGINE_PRESETS[dataset]
        crop_ratio = preset["crop_ratio"]
        a0, a1 = preset["row_anchor"]
        self.local_width = local_width

        self.model = _TRTModel(engine_path)
        self.W, self.H = int(image_size[0]), int(image_size[1])

        # Input geometry + grid sizes from the engine.
        self.train_height = int(self.model.input_shape[2])
        self.train_width = int(self.model.input_shape[3])
        loc_row_shape = self.model.engine.get_tensor_shape("loc_row")
        self.num_grid_row = int(loc_row_shape[1])
        self.num_cls_row = int(loc_row_shape[2])

        # Preprocessing feeds the bottom crop_ratio of the frame (full width). Row
        # anchors are fractions of the FULL image height, so class k sits at
        # original-image row yi[k] = row_anchor[k] * H.
        self.crop_rows = int(round(crop_ratio * self.H))
        self.yi = np.linspace(a0, a1, self.num_cls_row) * self.H

    # ---- preprocessing (GPU) ------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        device = torch.device("cuda")
        frame = frame[-self.crop_rows:, :, :]  # bottom crop_ratio of the image
        img = torch.from_numpy(frame).pin_memory().to(device, non_blocking=True, dtype=torch.float32)
        img = (img / 255.0).permute(2, 0, 1).unsqueeze(0)
        img = F.interpolate(img, size=(self.train_height, self.train_width),
                            mode="bilinear", align_corners=False)
        mean = torch.from_numpy(_MEAN).to(device).view(1, -1, 1, 1)
        std = torch.from_numpy(_STD).to(device).view(1, -1, 1, 1)
        return (img - mean) / std

    # ---- decoding (vectorized over all classes and both lanes) --------------
    def _decode_x(self, logits: np.ndarray) -> np.ndarray:
        """logits [num_grid, num_cls, L] -> original-image x [num_cls, L].

        Per (class, lane): softmax-weighted mean grid index over a window of
        +/-local_width around the argmax (+0.5 cell), scaled to image width.
        """
        G = self.num_grid_row
        amax = logits.argmax(0)                                  # [C, L]
        offsets = np.arange(-self.local_width, self.local_width + 1)
        raw = amax[None] + offsets[:, None, None]                # [win, C, L]
        in_range = (raw >= 0) & (raw < G)
        idx = np.clip(raw, 0, G - 1)

        w = np.take_along_axis(logits, idx, axis=0)              # [win, C, L]
        w = np.where(in_range, w, -np.inf)                       # ignore clipped neighbours
        w = np.exp(w - w.max(0, keepdims=True))
        w /= w.sum(0, keepdims=True)
        g = (w * idx).sum(0) + 0.5                               # [C, L]
        return g / (G - 1) * self.W

    def _pred2coords(self, pred: dict) -> LaneList:
        logits = pred["loc_row"][0][:, :, ROW_LANE_IDX]                   # [num_grid, num_cls, L]
        valid = pred["exist_row"][0].argmax(0)[:, ROW_LANE_IDX].astype(bool)  # [num_cls, L]
        x = self._decode_x(logits)                                        # [num_cls, L]

        lanes = []
        for j in range(len(ROW_LANE_IDX)):
            m = valid[:, j]
            if m.sum() <= self.num_cls_row / 2:
                lanes.append(EMPTY_LANE)
            else:
                lanes.append(np.column_stack([x[m, j], self.yi[m]]).astype(np.float32))
        return lanes

    # ---- public API ---------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> LaneList:
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != (self.W, self.H):
            frame_bgr = cv2.resize(frame_bgr, (self.W, self.H))
        outputs = self.model.infer_torch(self._preprocess(frame_bgr))
        pred = {name: outputs[name] for name in self.model.output_names}
        return self._pred2coords(pred)
