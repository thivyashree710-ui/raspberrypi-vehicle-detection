# Raspberry Pi Vehicle Detection and Counting System

An AI-based vehicle detection and counting application developed using Raspberry Pi 4, computer vision, and an IP camera stream. The project explores real-time vehicle detection, object tracking, vehicle counting, and automated reporting on an edge computing device.

## Project Overview

The system is designed to process camera footage, detect vehicles, track their movement, and support vehicle-counting and reporting workflows.

During development, different lightweight AI approaches were explored to balance detection performance and processing speed on Raspberry Pi hardware.

The Raspberry Pi implementation achieved approximately **4–5 FPS** during reported testing. Due to this hardware performance limitation, further development and performance optimisation were continued on the NVIDIA Jetson Xavier NX platform.

## Features

The repository contains the following application modules:

- Vehicle detection
- Vehicle tracking
- Application configuration
- Excel logging
- Video recording
- Utility functions for supporting the application workflow

RTSP camera integration, counting logic, report generation, snapshot capture, and other features should be verified against the current source code and configuration before deployment.

## Technology Stack

- **Programming language:** Python
- **Hardware platform:** Raspberry Pi 4
- **Computer vision:** OpenCV, if configured in the project environment
- **AI inference:** The project explored SSD MobileNet, TensorFlow Lite, and lightweight YOLO models
- **Reporting:** Excel logging
- **Video processing:** Camera/video processing and recording modules

The exact deployed model and dependency versions depend on the project configuration. Refer to the source files and installed environment for the definitive configuration.

## Repository Structure

```text
raspberrypi/
├── main.py
├── config.py
├── detector.py
├── tracker.py
├── utils.py
├── excel_logger.py
├── video_recorder.py
├── .gitignore
└── README.md
```

### Module Description

| File | Purpose |
|---|---|
| `main.py` | Main application entry point; verify execution flow in the source |
| `config.py` | Application configuration |
| `detector.py` | Vehicle/object detection logic |
| `tracker.py` | Object tracking logic |
| `utils.py` | Supporting utility functions |
| `excel_logger.py` | Excel logging and reporting functionality |
| `video_recorder.py` | Video recording functionality |
| `.gitignore` | Specifies files excluded from Git tracking |

## Performance

The Raspberry Pi implementation achieved approximately **4–5 FPS** during reported development testing.

This result reflects the processing limitations encountered on the Raspberry Pi platform. Actual FPS may vary depending on the AI model, input resolution, camera stream, and processing workload.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/thivyashree710-ui/raspberrypi-vehicle-detection.git
cd raspberrypi-vehicle-detection
```

### 2. Prepare the Python environment

Use a Python version compatible with the project's dependencies. A virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the dependencies required by the actual implementation. If a `requirements.txt` file is added to the repository, use:

```bash
pip install -r requirements.txt
```

**Note:** A dependency file is not currently listed in the repository structure above. Do not assume this command will work until `requirements.txt` is created.

### 4. Configure the application

Review `config.py` and update the required camera, model, output, and other application settings according to the source code.

Do not commit camera passwords, RTSP credentials, API keys, or other sensitive information to a public repository.

### 5. Run the application

The main application file is `main.py`. Confirm the required dependencies, model files, and configuration before running:

```bash
python main.py
```

The exact startup procedure may require additional setup depending on the model and camera configuration.

## Testing and Validation

The project was tested on Raspberry Pi hardware during development. The reported processing speed was approximately 4–5 FPS.

Further validation should cover:

- Camera-stream connectivity and stability
- Vehicle detection under different traffic conditions
- Tracking consistency and counting accuracy
- Excel report correctness
- Video recording reliability
- Long-duration operation
- CPU, memory, and processing performance

Test results should be documented only after the corresponding tests have been performed.

## Limitations

- Processing speed is limited by the Raspberry Pi hardware and selected AI model.
- Detection and tracking accuracy depend on camera quality, lighting, vehicle overlap, and model performance.
- The exact deployed model, dependency versions, and benchmark conditions should be confirmed from the project files.
- Further performance optimisation was pursued on NVIDIA Jetson Xavier NX.

## Future Improvements

- Improve inference speed and resource utilisation.
- Validate vehicle tracking and counting accuracy under simultaneous vehicle movement.
- Document the complete dependency and model setup.
- Add reproducible test results and benchmark methodology.
- Improve installation instructions and deployment reliability.

## Project Repository

GitHub: https://github.com/thivyashree710-ui/raspberrypi-vehicle-detection

## License

A license has not yet been specified. Add an appropriate license before distributing or reusing the project publicly.

## Acknowledgements

This project explores edge-based AI and computer vision for vehicle detection and counting on resource-constrained hardware.
