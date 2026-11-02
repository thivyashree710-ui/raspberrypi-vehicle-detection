"""
video_recorder.py - Handles MP4 video recording only.

Responsibilities:
  - Create VideoWriter on init
  - Accept annotated frames and write to disk
  - Release cleanly on exit
"""

from __future__ import annotations

import cv2
import numpy as np

from config import VIDEO_FILE, VIDEO_FPS, VIDEO_CODEC, VIDEO_WIDTH, VIDEO_HEIGHT


class VideoRecorder:
    """
    Wraps cv2.VideoWriter for clean MP4 recording.

    Usage:
        recorder = VideoRecorder()
        recorder.write(frame)
        recorder.release()
    """

    def __init__(self) -> None:
        import os
        os.makedirs(os.path.dirname(VIDEO_FILE), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self._writer = cv2.VideoWriter(
            VIDEO_FILE,
            fourcc,
            VIDEO_FPS,
            (VIDEO_WIDTH, VIDEO_HEIGHT),
        )

        if not self._writer.isOpened():
            raise RuntimeError(
                f"[VideoRecorder] Could not open VideoWriter for {VIDEO_FILE}"
            )

        print(f"[VideoRecorder] Recording to {VIDEO_FILE}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def write(self, frame: np.ndarray) -> None:
        """Write *frame* to the MP4 file. Resizes if needed."""
        h, w = frame.shape[:2]
        if w != VIDEO_WIDTH or h != VIDEO_HEIGHT:
            frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        self._writer.write(frame)

    def release(self) -> None:
        """Flush and close the video file."""
        if self._writer.isOpened():
            self._writer.release()
            print(f"[VideoRecorder] Saved {VIDEO_FILE}")
