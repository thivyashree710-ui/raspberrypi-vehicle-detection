"""
config.py - Central configuration for Vehicle Detection System.
All constants and settings live here; nothing is hardcoded elsewhere.
"""

import os
from datetime import datetime

# ======================
# CAMERA SETTINGS
# ======================
USERNAME = "admin"
PASSWORD = "cosmdumayil1965"
CAMERA_IP = "10.0.0.202"
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/Streaming/Channels/101"

# ======================
# MODEL SETTINGS
# ======================
MODEL_PATH = "detect.tflite"

# COCO class IDs mapped to display names
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "auto",
}

CONF_THRESHOLD = 0.50

# ======================
# ROI / ROAD BOUNDARIES
# ======================
# Trapezoid corners defining the drivable road area (x, y)
LEFT_BOTTOM  = (0,    720)
LEFT_TOP     = (650,  250)
RIGHT_TOP    = (1500, 250)
RIGHT_BOTTOM = (1150, 720)

# Horizontal counting line Y-coordinate
COUNTING_LINE_Y = 620

# Counting line X span
COUNTING_LINE_X_START = 250
COUNTING_LINE_X_END   = 1180

# ======================
# TRACKING SETTINGS
# ======================
MAX_DISTANCE        = 120   # px — max centroid distance for association
MAX_DISAPPEARED     = 5     # frames a track can vanish before removal
MIN_MOTION_DY       = 5     # px vertical motion required to count

# ======================
# VIDEO SETTINGS
# ======================
VIDEO_FPS           = 25
VIDEO_CODEC         = "mp4v"
VIDEO_WIDTH         = 1920
VIDEO_HEIGHT        = 1080

# ======================
# OUTPUT FOLDERS
# ======================
BASE_LOG_DIR   = "/home/test/VehicleLogs"
IMAGE_FOLDER   = os.path.join(BASE_LOG_DIR, "images")
VIDEO_FOLDER   = "/home/test/Videos"

_today = datetime.now().strftime("%Y%m%d")
_now   = datetime.now().strftime("%Y%m%d_%H%M%S")

EXCEL_FILE  = os.path.join(BASE_LOG_DIR, f"vehicle_log_{_today}.xlsx")
VIDEO_FILE  = os.path.join(VIDEO_FOLDER,  f"vehicle_{_now}.mp4")

# ======================
# EXCEL SETTINGS
# ======================
EXCEL_SAVE_INTERVAL = 10   # Save workbook every N vehicles

# Column widths (letter -> width)
EXCEL_COL_WIDTHS = {
    "A": 22,   # Timestamp
    "B": 12,   # Vehicle_ID
    "C": 18,   # Vehicle_Type
    "D": 12,   # Direction
    "E": 14,   # Confidence
    "F": 28,   # Vehicle_Image
    "G": 12,   # Total_Count
}

EXCEL_IMAGE_WIDTH  = 160
EXCEL_IMAGE_HEIGHT = 120
EXCEL_ROW_HEIGHT   = 95    # points

# ======================
# DISPLAY COLOURS (BGR)
# ======================
COLOR_BOX       = (0,   255,   0)
COLOR_CENTROID  = (0,     0, 255)
COLOR_LINE_ROI  = (0,   255,   0)
COLOR_LINE_CNT  = (255,   0,   0)
COLOR_TEXT_HUD  = (0,   255, 255)
COLOR_LABEL     = (0,   255,   0)
