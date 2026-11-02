"""
excel_logger.py - Excel report generation.

Responsibilities:
  - Create / load workbook once on startup
  - Append one row per counted vehicle
  - Embed cropped vehicle image directly in the cell
  - Adjust row height and column widths
  - Save every EXCEL_SAVE_INTERVAL vehicles and on exit
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Tuple

import cv2
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as XLImage

from config import (
    EXCEL_FILE, IMAGE_FOLDER,
    EXCEL_COL_WIDTHS, EXCEL_IMAGE_WIDTH, EXCEL_IMAGE_HEIGHT,
    EXCEL_ROW_HEIGHT, EXCEL_SAVE_INTERVAL,
)


class ExcelLogger:
    """
    Thread-safe-enough (single-writer) Excel logger.

    Usage:
        logger = ExcelLogger()
        logger.log(frame, vehicle_id, vehicle_type, direction,
                   confidence, total_count, bbox)
        logger.close()
    """

    HEADERS = [
        "Timestamp", "Vehicle_ID", "Vehicle_Type",
        "Direction", "Confidence", "Vehicle_Image", "Total_Count",
    ]

    def __init__(self) -> None:
        os.makedirs(IMAGE_FOLDER, exist_ok=True)
        os.makedirs(os.path.dirname(EXCEL_FILE), exist_ok=True)

        self._vehicle_counter = 0   # counts logged vehicles (for save cadence)

        # Load existing workbook or create fresh one
        if os.path.exists(EXCEL_FILE):
            self._wb = load_workbook(EXCEL_FILE)
            self._ws = self._wb["Vehicle Log"]
        else:
            self._wb = Workbook()
            self._ws = self._wb.active
            self._ws.title = "Vehicle Log"
            self._ws.append(self.HEADERS)
            self._apply_column_widths()
            self._wb.save(EXCEL_FILE)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def log(
        self,
        frame:        np.ndarray,
        vehicle_id:   int,
        vehicle_type: str,
        direction:    str,
        confidence:   float,
        total_count:  int,
        bbox:         Tuple[int, int, int, int],   # (xmin, ymin, xmax, ymax)
    ) -> None:
        """Append a row and embed the cropped vehicle image."""
        timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_path = self._save_crop(frame, vehicle_id, bbox)

        # Append data row (image cell left blank — we embed below)
        self._ws.append([
            timestamp,
            vehicle_id,
            vehicle_type,
            direction,
            f"{confidence:.2%}",
            "",            # placeholder for the image column
            total_count,
        ])

        # Embed image into column F of the new row
        row_num = self._ws.max_row
        try:
            img        = XLImage(image_path)
            img.width  = EXCEL_IMAGE_WIDTH
            img.height = EXCEL_IMAGE_HEIGHT
            self._ws.add_image(img, f"F{row_num}")
        except Exception as exc:
            print(f"[ExcelLogger] Image embed failed: {exc}")

        # Set row height so the image is fully visible
        self._ws.row_dimensions[row_num].height = EXCEL_ROW_HEIGHT

        self._vehicle_counter += 1

        # Periodic save
        if self._vehicle_counter % EXCEL_SAVE_INTERVAL == 0:
            self._save()

    def close(self) -> None:
        """Final save on exit."""
        self._save()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            self._wb.save(EXCEL_FILE)
        except Exception as exc:
            print(f"[ExcelLogger] Save failed: {exc}")

    def _apply_column_widths(self) -> None:
        for col_letter, width in EXCEL_COL_WIDTHS.items():
            self._ws.column_dimensions[col_letter].width = width

    def _save_crop(
        self,
        frame:      np.ndarray,
        vehicle_id: int,
        bbox:       Tuple[int, int, int, int],
        margin:     int = 10,
    ) -> str:
        """Crop the vehicle from *frame*, save to disk, return path."""
        H, W = frame.shape[:2]
        xmin, ymin, xmax, ymax = bbox

        x1 = max(0, xmin - margin)
        y1 = max(0, ymin - margin)
        x2 = min(W,  xmax + margin)
        y2 = min(H,  ymax + margin)

        crop = frame[y1:y2, x1:x2]
        path = os.path.join(IMAGE_FOLDER, f"{vehicle_id}.jpg")
        cv2.imwrite(path, crop)
        return path
