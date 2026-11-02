"""
main.py - Vehicle Detection & Counting System — entry point.

Coordinates all modules:
  Detector        → inference + tracking + counting
  ExcelLogger     → Excel report with embedded images
  VideoRecorder   → MP4 recording
  FPSCounter/HUD  → performance overlay
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"   # Needed on headless Linux / Raspberry Pi

import time
import cv2

from detector       import Detector
from excel_logger   import ExcelLogger
from video_recorder import VideoRecorder
from utils          import FPSCounter, draw_hud


def main() -> None:
    print("[Main] Initialising modules …")

    detector  = Detector()
    logger    = ExcelLogger()
    recorder  = VideoRecorder()
    fps_counter = FPSCounter(window=10)

    print("[Main] Starting detection loop. Press 'q' to quit.")

    try:
        while True:
            # ---- Read frame ----
            frame = detector.read_frame()
            if frame is None:
                time.sleep(0.005)   # wait for capture thread
                continue

            # ---- Run detection / tracking / counting ----
            annotated_frame, counted_vehicles = detector.detect(frame)

            # ---- Log newly counted vehicles ----
            for cv_event in counted_vehicles:
                logger.log(
                    frame        = frame,           # original (unannotated) for clean crop
                    vehicle_id   = cv_event.track_id,
                    vehicle_type = cv_event.vehicle_type,
                    direction    = cv_event.direction,
                    confidence   = cv_event.confidence,
                    total_count  = cv_event.total_count,
                    bbox         = cv_event.bbox,
                )
                print(
                    f"[Count] #{cv_event.track_id} {cv_event.vehicle_type} "
                    f"{cv_event.direction}  total={cv_event.total_count}"
                )

            # ---- HUD overlay ----
            fps = fps_counter.tick()
            draw_hud(annotated_frame, fps, detector.vehicle_count)

            # ---- Record frame ----
            recorder.write(annotated_frame)

            # ---- Display ----
            cv2.imshow("Vehicle Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Main] Quit signal received.")
                break

    except KeyboardInterrupt:
        print("[Main] Interrupted by user.")

    finally:
        print("[Main] Cleaning up …")
        recorder.release()
        logger.close()
        cv2.destroyAllWindows()
        print("[Main] Done.")


if __name__ == "__main__":
    main()
