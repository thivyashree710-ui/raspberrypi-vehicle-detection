"""
tracker.py - Stable centroid tracker with multi-frame persistence.

Each track stores:
  - object_id       : unique integer
  - current_center  : (cx, cy) this frame
  - previous_center : (cx, cy) last frame
  - bbox            : (xmin, ymin, xmax, ymax)
  - vehicle_type    : str
  - confidence      : float
  - direction       : "UP" | "DOWN" | "UNKNOWN"
  - counted         : bool
  - disappeared     : frames since last match
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import MAX_DISTANCE, MAX_DISAPPEARED


@dataclass
class Track:
    """Single tracked object."""
    object_id:       int
    current_center:  Tuple[int, int]
    previous_center: Optional[Tuple[int, int]] = None
    bbox:            Tuple[int, int, int, int]  = (0, 0, 0, 0)
    vehicle_type:    str                        = "unknown"
    confidence:      float                      = 0.0
    direction:       str                        = "UNKNOWN"
    counted:         bool                       = False
    disappeared:     int                        = 0

    def update_direction(self) -> None:
        """Compute direction from previous → current centroid."""
        if self.previous_center is None:
            return
        dy = self.current_center[1] - self.previous_center[1]
        if dy > 0:
            self.direction = "DOWN"
        elif dy < 0:
            self.direction = "UP"
        # else keep existing direction


class CentroidTracker:
    """
    Associates new detections to existing tracks via nearest-centroid matching.

    Tracks survive up to MAX_DISAPPEARED frames without a detection,
    which keeps IDs stable through brief occlusions.
    """

    def __init__(
        self,
        max_distance: int = MAX_DISTANCE,
        max_disappeared: int = MAX_DISAPPEARED,
    ) -> None:
        self._next_id       = 0
        self.tracks:        Dict[int, Track] = {}
        self.max_distance   = max_distance
        self.max_disappeared = max_disappeared

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Tuple[int, int, str, int, int, int, int, float]],
    ) -> Dict[int, Track]:
        """
        Match detections to existing tracks and return the updated track dict.

        Parameters
        ----------
        detections : list of (cx, cy, vehicle_type, xmin, ymin, xmax, ymax, confidence)

        Returns
        -------
        dict mapping object_id -> Track
        """
        # ---- No detections: age every track ----
        if not detections:
            self._age_all()
            self._prune()
            return self.tracks

        # ---- No existing tracks: register all ----
        if not self.tracks:
            for det in detections:
                self._register(det)
            return self.tracks

        # ---- Hungarian-style greedy nearest-centroid matching ----
        track_ids    = list(self.tracks.keys())
        track_centers = np.array(
            [self.tracks[tid].current_center for tid in track_ids],
            dtype=np.float32,
        )
        det_centers   = np.array(
            [(d[0], d[1]) for d in detections],
            dtype=np.float32,
        )

        # Pairwise Euclidean distance matrix [tracks × detections]
        D = self._distance_matrix(track_centers, det_centers)

        # Greedily match minimum-distance pairs
        matched_tracks = set()
        matched_dets   = set()

        # Sort all (dist, t_idx, d_idx) pairs
        rows_cols = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
        for t_idx, d_idx in rows_cols:
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            if D[t_idx, d_idx] > self.max_distance:
                break  # remaining distances are all ≥ this (sorted)
            tid = track_ids[t_idx]
            self._associate(tid, detections[d_idx])
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

        # Age unmatched tracks
        for t_idx, tid in enumerate(track_ids):
            if t_idx not in matched_tracks:
                self.tracks[tid].disappeared += 1

        # Register unmatched detections as new tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self._register(det)

        self._prune()
        return self.tracks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register(
        self,
        det: Tuple[int, int, str, int, int, int, int, float],
    ) -> None:
        cx, cy, vtype, xmin, ymin, xmax, ymax, conf = det
        track = Track(
            object_id      = self._next_id,
            current_center = (cx, cy),
            bbox           = (xmin, ymin, xmax, ymax),
            vehicle_type   = vtype,
            confidence     = conf,
        )
        self.tracks[self._next_id] = track
        self._next_id += 1

    def _associate(
        self,
        tid: int,
        det: Tuple[int, int, str, int, int, int, int, float],
    ) -> None:
        cx, cy, vtype, xmin, ymin, xmax, ymax, conf = det
        t = self.tracks[tid]
        t.previous_center = t.current_center
        t.current_center  = (cx, cy)
        t.bbox            = (xmin, ymin, xmax, ymax)
        t.vehicle_type    = vtype
        t.confidence      = conf
        t.disappeared     = 0
        t.update_direction()

    def _age_all(self) -> None:
        for t in self.tracks.values():
            t.disappeared += 1

    def _prune(self) -> None:
        dead = [tid for tid, t in self.tracks.items()
                if t.disappeared > self.max_disappeared]
        for tid in dead:
            del self.tracks[tid]

    @staticmethod
    def _distance_matrix(
        a: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        """Return (len(a), len(b)) Euclidean distance matrix."""
        # ||a - b||² = ||a||² + ||b||² - 2 a·bᵀ
        a2 = np.sum(a ** 2, axis=1, keepdims=True)  # (m, 1)
        b2 = np.sum(b ** 2, axis=1, keepdims=True)  # (n, 1)
        D2 = a2 + b2.T - 2.0 * (a @ b.T)
        D2 = np.maximum(D2, 0.0)                    # numerical safety
        return np.sqrt(D2)
