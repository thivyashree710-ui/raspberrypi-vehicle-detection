"""
utils.py - Shared helper utilities.

Provides:
  - FPSCounter  : rolling FPS calculation
  - draw_hud()  : overlay FPS, count, timestamp on a frame
"""

from __future__ import annotations

import time
from datetime import datetime

import cv2
import numpy as np

from config import COLOR_TEXT_HUD


class FPSCounter:
    """Rolling FPS calculator over a fixed number of frames."""

    def __init__(self, window: int = 10) -> None:
        self._window     = window
        self._frame_count = 0
        self._start       = time.time()
        self.fps          = 0.0

    def tick(self) -> float:
        """Call once per frame; returns current FPS."""
        self._frame_count += 1
        if self._frame_count >= self._window:
            elapsed          = time.time() - self._start
            self.fps         = self._frame_count / elapsed
            self._frame_count = 0
            self._start       = time.time()
        return self.fps


def draw_hud(
    frame:         np.ndarray,
    fps:           float,
    vehicle_count: int,
) -> np.ndarray:
    """
    Draw heads-up display onto *frame* in-place.

    Shows FPS, vehicle count, and current timestamp.
    Returns the same frame for convenience.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        (f"FPS: {fps:.1f}",                (20, 40)),
        (f"Vehicle Count: {vehicle_count}", (20, 80)),
        (timestamp,                         (20, 120)),
    ]

    for text, pos in lines:
        cv2.putText(
            frame, text, pos,
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            COLOR_TEXT_HUD, 2, cv2.LINE_AA,
        )

    return frame
