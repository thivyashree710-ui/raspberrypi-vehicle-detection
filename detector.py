"""
detector.py - Camera capture, TFLite inference, ROI filtering,
              vehicle tracking, and counting.

No Excel code. No VideoWriter code. No openpyxl imports.
Returns detection results to main.py via detect().
"""

from __future__ import annotations

import threading
from queue import Queue
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

from config import (
    RTSP_URL, MODEL_PATH,
    VEHICLE_CLASSES, CONF_THRESHOLD,
    LEFT_BOTTOM, LEFT_TOP, RIGHT_TOP, RIGHT_BOTTOM,
    COUNTING_LINE_Y, COUNTING_LINE_X_START, COUNTING_LINE_X_END,
    MIN_MOTION_DY,
    COLOR_BOX, COLOR_CENTROID, COLOR_LINE_ROI, COLOR_LINE_CNT, COLOR_LABEL,
)
from tracker import CentroidTracker, Track


@dataclass
class CountedVehicle:
    """Emitted once per counted crossing event."""
    track_id:     int
    vehicle_type: str
    direction:    str
    confidence:   float
    bbox:         Tuple[int, int, int, int]   # (xmin, ymin, xmax, ymax)
    total_count:  int


class Detector:
    """
    Encapsulates the full detection pipeline:
      1. Threaded RTSP capture
      2. TFLite inference
      3. ROI filtering
      4. Centroid tracking
      5. Line-crossing counting
      6. Frame annotation
    """

    def __init__(self) -> None:
        # ---- Load TFLite model ----
        self._interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self._interpreter.allocate_tensors()

        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        self._model_h        = self._input_details[0]['shape'][1]
        self._model_w        = self._input_details[0]['shape'][2]
        self._floating_model = self._input_details[0]['dtype'] == np.float32

        # Pre-allocate reusable input buffer (avoids per-frame malloc)
        self._input_buf = np.zeros(
            (1, self._model_h, self._model_w, 3), dtype=np.float32
        )

        # ---- Tracker ----
        self._tracker      = CentroidTracker()
        self._vehicle_count = 0

        # ---- Threaded frame queue ----
        self._frame_queue = Queue(maxsize=2)
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._capture_thread.start()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read_frame(self) -> Optional[np.ndarray]:
        """Return the latest camera frame, or None if not yet available."""
        if self._frame_queue.empty():
            return None
        return self._frame_queue.get()

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[CountedVehicle]]:
        """
        Run inference + tracking on *frame*.

        Parameters
        ----------
        frame : BGR numpy array (original resolution)

        Returns
        -------
        annotated_frame  : frame with boxes, IDs, lines drawn
        counted_vehicles : list of CountedVehicle for newly-counted crossings
        """
        H, W = frame.shape[:2]

        # ---- Inference ----
        raw_detections = self._run_inference(frame, H, W)

        # ---- Tracker update ----
        tracks = self._tracker.update(raw_detections)

        # ---- Count line crossings ----
        newly_counted: List[CountedVehicle] = []
        for tid, track in tracks.items():
            if track.counted:
                continue
            if track.previous_center is None:
                continue

            py = track.previous_center[1]
            cy = track.current_center[1]

            # Must cross the counting line
            crossed_down = py < COUNTING_LINE_Y <= cy
            crossed_up   = py > COUNTING_LINE_Y >= cy

            if not (crossed_down or crossed_up):
                continue

            # Vertical motion must be meaningful
            if abs(cy - py) < MIN_MOTION_DY:
                continue

            direction = "DOWN" if crossed_down else "UP"
            track.direction = direction
            track.counted   = True
            self._vehicle_count += 1

            newly_counted.append(CountedVehicle(
                track_id     = tid,
                vehicle_type = track.vehicle_type,
                direction    = direction,
                confidence   = track.confidence,
                bbox         = track.bbox,
                total_count  = self._vehicle_count,
            ))

        # ---- Annotate frame ----
        annotated = self._annotate(frame, tracks)

        return annotated, newly_counted

    @property
    def vehicle_count(self) -> int:
        return self._vehicle_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Background thread: reads RTSP frames and pushes to queue."""
        cap = cv2.VideoCapture(RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.open(RTSP_URL)   # reconnect on failure
                continue

            # Keep only the freshest frame; drop stale one if queue full
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except Exception:
                    pass

            self._frame_queue.put(frame)

    def _run_inference(
        self, frame: np.ndarray, H: int, W: int
    ) -> List[Tuple[int, int, str, int, int, int, int, float]]:
        """
        Pre-process, invoke TFLite, post-process detections.
        Returns list of (cx, cy, vtype, xmin, ymin, xmax, ymax, score).
        """
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._model_w, self._model_h))

        # Reuse pre-allocated buffer
        np.copyto(self._input_buf[0], resized.astype(np.float32))
        if self._floating_model:
            self._input_buf[0] = (self._input_buf[0] - 127.5) / 127.5

        self._interpreter.set_tensor(
            self._input_details[0]['index'], self._input_buf
        )
        self._interpreter.invoke()

        boxes   = self._interpreter.get_tensor(self._output_details[0]['index'])[0]
        classes = self._interpreter.get_tensor(self._output_details[1]['index'])[0]
        scores  = self._interpreter.get_tensor(self._output_details[2]['index'])[0]

        detections: List[Tuple] = []

        for i in range(len(scores)):
            if scores[i] < CONF_THRESHOLD:
                continue

            cls = int(classes[i])
            if cls not in VEHICLE_CLASSES:
                continue

            ymin_r, xmin_r, ymax_r, xmax_r = boxes[i]
            xmin = int(xmin_r * W)
            xmax = int(xmax_r * W)
            ymin = int(ymin_r * H)
            ymax = int(ymax_r * H)

            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            # ---- ROI filtering ----
            left_x = int(np.interp(
                cy,
                [LEFT_TOP[1],    LEFT_BOTTOM[1]],
                [LEFT_TOP[0],    LEFT_BOTTOM[0]],
            ))
            right_x = int(np.interp(
                cy,
                [RIGHT_TOP[1],   RIGHT_BOTTOM[1]],
                [RIGHT_TOP[0],   RIGHT_BOTTOM[0]],
            ))

            if cx < left_x or cx > right_x:
                continue

            detections.append((
                cx, cy, VEHICLE_CLASSES[cls],
                xmin, ymin, xmax, ymax,
                float(scores[i]),
            ))

        return detections

    def _annotate(
        self, frame: np.ndarray, tracks: Dict[int, "Track"]
    ) -> np.ndarray:
        """Draw bounding boxes, IDs, directions, and ROI/counting lines."""
        out = frame.copy()

        # ---- Per-vehicle boxes ----
        for tid, track in tracks.items():
            if track.disappeared > 0:
                continue           # skip stale tracks

            xmin, ymin, xmax, ymax = track.bbox
            cx, cy = track.current_center
            label = (
                f"#{tid} {track.vehicle_type} {track.direction} "
                f"{track.confidence:.0%}"
            )

            cv2.rectangle(out, (xmin, ymin), (xmax, ymax), COLOR_BOX, 2)
            cv2.putText(
                out, label, (xmin, ymin - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 2,
            )
            cv2.circle(out, (cx, cy), 4, COLOR_CENTROID, -1)

        # ---- Road ROI boundary ----
        cv2.line(out, LEFT_BOTTOM,  LEFT_TOP,    COLOR_LINE_ROI, 3)
        cv2.line(out, LEFT_TOP,     RIGHT_TOP,   COLOR_LINE_ROI, 3)
        cv2.line(out, RIGHT_TOP,    RIGHT_BOTTOM, COLOR_LINE_ROI, 3)

        # ---- Counting line ----
        cv2.line(
            out,
            (COUNTING_LINE_X_START, COUNTING_LINE_Y),
            (COUNTING_LINE_X_END,   COUNTING_LINE_Y),
            COLOR_LINE_CNT, 3,
        )

        return out
