"""
face_detection.py — Face detector interface for DogDayDiffuser.

Provides:
  - FaceInfo   dataclass with detection results
  - FaceDetector   abstract base
  - HaarFaceDetector   OpenCV Haar cascade (CPU, no extra deps)
  - OpenVINOFaceDetector   Intel NCS2 / OpenVINO backend (optional)
  - build_detector()   factory function

Detection does not need to run every frame.  The caller (main loop) decides
the detection interval and passes cached FaceInfo between detections.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Path to OpenCV's bundled Haar cascade for frontal faces
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FaceInfo:
    """Result of a single face detection pass."""

    # Bounding box in the coordinate space of the *processed* frame
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    # Normalised centre (0.0 – 1.0) relative to the frame dimensions
    cx_norm: float = 0.5
    cy_norm: float = 0.5

    # Approximate face size relative to frame area (0.0 – 1.0)
    size_norm: float = 0.0

    # Whether a face was actually found
    detected: bool = False

    @classmethod
    def from_rect(cls, x: int, y: int, w: int, h: int,
                  frame_w: int, frame_h: int) -> "FaceInfo":
        """Build a FaceInfo from a bounding-box rect and frame dimensions."""
        cx = x + w // 2
        cy = y + h // 2
        face_area = w * h
        frame_area = frame_w * frame_h
        return cls(
            x=x, y=y, w=w, h=h,
            cx_norm=cx / frame_w,
            cy_norm=cy / frame_h,
            size_norm=face_area / max(frame_area, 1),
            detected=True,
        )


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class FaceDetector(ABC):
    """Abstract base class for face detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[FaceInfo]:
        """Detect the most prominent face in *frame* (BGR uint8).

        Returns a :class:`FaceInfo` if a face is found, or ``None``.
        """

    def close(self) -> None:
        """Release any resources held by the detector (override if needed)."""


# ---------------------------------------------------------------------------
# OpenCV Haar cascade detector
# ---------------------------------------------------------------------------


class HaarFaceDetector(FaceDetector):
    """OpenCV Haar cascade frontal-face detector.

    This is the default CPU-only fallback.  It works without any optional
    dependencies.
    """

    def __init__(self, scale_factor: float = 1.2, min_neighbours: int = 4,
                 min_size: tuple = (30, 30)):
        if not os.path.isfile(_HAAR_PATH):
            raise FileNotFoundError(
                f"Haar cascade XML not found at {_HAAR_PATH}. "
                "Reinstall opencv-python."
            )
        self._cascade = cv2.CascadeClassifier(_HAAR_PATH)
        self._scale_factor = scale_factor
        self._min_neighbours = min_neighbours
        self._min_size = min_size
        logger.info("HaarFaceDetector ready (scale=%.1f, minN=%d)",
                    scale_factor, min_neighbours)

    def detect(self, frame: np.ndarray) -> Optional[FaceInfo]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbours,
            minSize=self._min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return None

        # Return the largest detected face
        h_frame, w_frame = frame.shape[:2]
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return FaceInfo.from_rect(x, y, w, h, w_frame, h_frame)


# ---------------------------------------------------------------------------
# OpenVINO / NCS2 detector  (optional)
# ---------------------------------------------------------------------------


class OpenVINOFaceDetector(FaceDetector):
    """Face detector backed by Intel OpenVINO inference engine.

    Uses the *face-detection-retail-0004* model which is part of the
    OpenVINO Open Model Zoo.  Falls back gracefully if the model or the
    runtime is unavailable.

    Args:
        model_xml: Path to the model .xml file.
        model_bin: Path to the model .bin file (auto-inferred if None).
        device:    Target device, e.g. ``"MYRIAD"`` for NCS2 or ``"CPU"``.
        threshold: Confidence threshold (0–1).
    """

    def __init__(self, model_xml: str,
                 model_bin: Optional[str] = None,
                 device: str = "MYRIAD",
                 threshold: float = 0.6):
        try:
            from openvino.runtime import Core  # type: ignore
        except ImportError:
            raise ImportError(
                "openvino package is not installed.  "
                "Install it with: pip install openvino"
            )

        if model_bin is None:
            model_bin = model_xml.replace(".xml", ".bin")

        if not os.path.isfile(model_xml):
            raise FileNotFoundError(f"OpenVINO model XML not found: {model_xml}")
        if not os.path.isfile(model_bin):
            raise FileNotFoundError(f"OpenVINO model BIN not found: {model_bin}")

        core = Core()
        model = core.read_model(model=model_xml, weights=model_bin)
        compiled = core.compile_model(model=model, device_name=device)
        self._infer = compiled.create_infer_request()
        self._input_layer = compiled.input(0)
        self._output_layer = compiled.output(0)
        self._threshold = threshold

        # Expected input shape: [1, C, H, W]
        _, _, self._in_h, self._in_w = self._input_layer.shape
        logger.info("OpenVINOFaceDetector ready on device=%s (%.0f×%.0f)",
                    device, self._in_w, self._in_h)

    def detect(self, frame: np.ndarray) -> Optional[FaceInfo]:
        h_frame, w_frame = frame.shape[:2]

        # Pre-process: resize and convert to NCHW float32
        blob = cv2.resize(frame, (self._in_w, self._in_h))
        blob = blob.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        self._infer.infer({self._input_layer: blob})
        detections = self._infer.get_tensor(self._output_layer).data[0][0]

        best: Optional[FaceInfo] = None
        best_conf = 0.0

        for det in detections:
            # det: [image_id, label, conf, x1, y1, x2, y2]  (coords are 0–1)
            conf = float(det[2])
            if conf < self._threshold or conf < best_conf:
                continue

            x1 = int(det[3] * w_frame)
            y1 = int(det[4] * h_frame)
            x2 = int(det[5] * w_frame)
            y2 = int(det[6] * h_frame)
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)

            best = FaceInfo.from_rect(x1, y1, w, h, w_frame, h_frame)
            best_conf = conf

        return best


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_detector(use_openvino: bool = False,
                   openvino_model_xml: Optional[str] = None,
                   openvino_device: str = "MYRIAD") -> FaceDetector:
    """Return an appropriate FaceDetector, with graceful fallback.

    If *use_openvino* is True but OpenVINO is unavailable (missing package,
    missing model, or NCS2 not attached) the function logs a warning and
    falls back to the Haar cascade detector.
    """
    if use_openvino:
        try:
            xml = openvino_model_xml or "models/face-detection-retail-0004.xml"
            detector = OpenVINOFaceDetector(xml, device=openvino_device)
            logger.info("Using OpenVINO face detector")
            return detector
        except Exception as exc:
            logger.warning(
                "OpenVINO face detector unavailable (%s). "
                "Falling back to Haar cascade.",
                exc,
            )

    detector = HaarFaceDetector()
    logger.info("Using Haar cascade face detector")
    return detector
