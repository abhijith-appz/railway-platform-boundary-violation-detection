# 🚆 Railway Platform Boundary Violation Detection

A computer-vision prototype that analyses railway-platform video and flags potential pedestrian entry into a configured track-side danger zone. It uses YOLO-based person detection, tracking, foot-point geometry, and a configurable polygon to identify possible boundary violations. A Streamlit dashboard lets users upload a video, configure the danger zone, run analysis, and review annotated results.

> **Project status:** Academic / research prototype. This system is not certified for railway operations and must not be used as the sole safety mechanism. Validate it on representative footage and have a qualified human review every alert.

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [How it works](#how-it-works)
- [Technology stack](#technology-stack)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run the web dashboard](#run-the-web-dashboard)
- [Run detection from the command line](#run-detection-from-the-command-line)
- [Configure the danger zone](#configure-the-danger-zone)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Limitations and safety](#limitations-and-safety)
- [Future improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Overview

Railway platforms can become hazardous when a passenger enters the track area or moves beyond the platform safety boundary. This project explores how computer vision can assist monitoring by analysing recorded video or supported camera streams and identifying people whose estimated foot point enters a user-defined danger zone.

The danger zone is a polygon in the camera image. The system detects people, estimates the bottom-centre (foot) point of each person’s bounding box, and checks whether that point lies inside the polygon. A violation is raised only after the configured number of consecutive frames, helping reduce alerts caused by one-frame detection noise.

The system can generate an annotated video and incident records, including snapshots. An optional webhook can be configured for integration experiments.

## Features

- **YOLO person detection** using Ultralytics model weights.
- **Person tracking** to associate detections across video frames.
- **Configurable danger-zone polygon** using normalized coordinates from `0.0` to `1.0`.
- **Consecutive-frame confirmation** to reduce transient false positives.
- **Streamlit dashboard** for uploading videos, adjusting detection settings, previewing the zone, and running analysis.
- **Annotated video output** for reviewing detections and potential violations.
- **Incident records** in CSV format and snapshot images.
- **Optional webhook URL** for testing external alert integrations.
- **Polygon selection utility** to define a zone from a video frame.
- **Optional ONNX Runtime backend** for supported inference setups.
- **Logic test** for core boundary-checking behavior.

## How it works

1. **Video input:** Load an uploaded video or a supported video source.
2. **Person detection:** Run a YOLO model on each processed frame.
3. **Tracking:** Associate detections with person IDs where tracking is available.
4. **Foot-point estimation:** Use the bottom-centre of a person’s bounding box as an approximation of their ground-contact point.
5. **Zone test:** Check whether the foot point falls inside the configured danger-zone polygon.
6. **Confirmation:** Require the point to remain in the zone for a configured number of frames before generating an event.
7. **Reporting:** Save an annotated video, CSV incident records, and snapshots; optionally send a webhook notification.

This is a geometric zone-entry prototype. It does not inherently understand the meaning of a yellow line, determine intent, or prove that a person has crossed a physical line. The polygon must be calibrated for the particular camera view.

## Technology stack

- **Python** — application and detection logic
- **Ultralytics YOLO** — object/person detection and tracking
- **OpenCV** — video reading, drawing, and writing
- **NumPy** — coordinate and polygon operations
- **Streamlit** — interactive web dashboard
- **Pandas** — incident tables and CSV export
- **Requests** — optional webhook requests
- **ONNX Runtime** — optional ONNX inference backend

## Repository structure

```text
railway_boundary/
├── app.py                 # Streamlit web dashboard
├── detector.py            # Main detection, tracking, and boundary logic
├── onnx_detector.py       # Optional ONNX inference backend
├── yellow_line.py         # Experimental yellow-line detection utilities
├── select_zone.py         # Tool for selecting a polygon zone from video
├── test_logic.py          # Basic boundary-logic test
├── zone.example.json      # Example zone configuration
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Files excluded from Git
```

Generated output folders, local videos, snapshots, model weights, and Python cache files should generally not be committed unless they are small, licensed for redistribution, and intentionally included as examples.

## Requirements

- Python 3.10 or newer is recommended; use a Python version supported by the installed PyTorch/Ultralytics packages.
- `pip` and `venv`
- FFmpeg is optional but useful for checking and converting video files.
- CPU can be used for testing; a compatible NVIDIA GPU can improve inference speed.
- Internet access may be needed on first run to download the selected YOLO weights.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/railway-boundary-violation-detection.git
cd railway-boundary-violation-detection
```

Create and activate a virtual environment.

**Ubuntu / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch or Ultralytics installation fails, install a PyTorch build compatible with your Python version and hardware, then install the remaining dependencies. For a CPU-only machine, use the official PyTorch installation selector to choose the appropriate command.

## Run the web dashboard

Start Streamlit from the project directory:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

1. Upload a supported video (`.mp4`, `.avi`, `.mov`, or `.mkv`).
2. Select the YOLO model and set the confidence, inference size, confirmation-frame count, and cooldown.
3. Inspect the danger-zone preview.
4. Edit the `danger_zone` JSON coordinates to match the track-side area for this camera.
5. Select **Analyse video**.
6. Review the annotated video and any reported events. Download the CSV if events are found.

The dashboard’s default polygon is only a starting point for the example camera perspective. It must be checked and adjusted for every new video/camera. Do not assume the default zone is correct for other stations or camera angles.

## Run detection from the command line

### Select a zone

```bash
python select_zone.py --video platform.mp4 --out zone.json
```

Follow the utility’s on-screen instructions to click the polygon corners and save the zone.

### Analyse a video using the saved zone

```bash
python detector.py --video platform.mp4 --zone zone.json --show
```

### Try automatic yellow-line detection (experimental)

```bash
python detector.py --video platform.mp4 --auto-line --track-side left --show
```

Use `--track-side right` if the track is on the right side of the detected line in the image. Automatic colour-based line detection can fail with shadows, poor lighting, worn markings, occlusion, or similarly coloured surfaces; review its results carefully.

### ONNX backend (optional)

Export a compatible model to ONNX using Ultralytics:

```bash
yolo export model=yolov8s.pt format=onnx imgsz=640
```

Then, if supported by the current CLI and model configuration:

```bash
python detector.py --video platform.mp4 --auto-line --backend onnx --weights yolov8s.onnx
```

Check `python detector.py --help` if command-line options differ in your checked-out version.

### Run the logic test

```bash
python test_logic.py
```

This tests boundary logic only; it does not measure real-world detection accuracy or validate the system for operational safety.

## Configure the danger zone

The polygon is expressed as normalized `[x, y]` coordinates:

- `x = 0.0` is the left edge; `x = 1.0` is the right edge.
- `y = 0.0` is the top edge; `y = 1.0` is the bottom edge.

Example format:

```json
{
  "normalized": true,
  "danger_zone": [
    [0.20, 0.00],
    [0.56, 0.00],
    [0.72, 1.00],
    [0.34, 1.00]
  ]
}
```

The example above is illustrative, not a universal railway-platform polygon. The polygon should cover the intended track-side danger area in the image and follow the perspective of the platform edge. Confirm the direction of the track and the camera view before testing.

For a video with width `W` and height `H`, a normalized point converts approximately to pixels using `pixel_x = x * W` and `pixel_y = y * H`.

## Outputs

The detector writes results to its configured output directory (the default CLI output is `output/`). Typical artifacts include:

- `annotated.mp4` — processed video with visual overlays
- `violations.csv` — recorded violation events
- `snapshots/` — images saved for reported events

The dashboard may use a temporary working directory during a session, so download or copy any results you need to retain.

If an MP4 does not play in a browser, try converting it with FFmpeg:

```bash
ffmpeg -i annotated.mp4 -c:v libx264 -pix_fmt yuv420p annotated_h264.mp4
```

## Troubleshooting

### The dashboard says “No confirmed boundary violation detected”

- Confirm that the uploaded video contains frames that can be read.
- Check whether green person-detection boxes appear in the annotated video.
- If there are no boxes, try a suitable model, a lower confidence threshold, or a larger inference size; performance depends on the footage and hardware.
- If people are detected, inspect the foot-point markers and confirm they enter the polygon.
- Redraw the polygon to match the actual track-side area in that camera view.
- Reduce confirmation frames temporarily for debugging, then increase them as needed to reduce false alarms.
- Check the terminal for model-loading, tracking, or video-writing errors.

“No confirmed violation” means the configured rule did not trigger. It does **not** prove that the scene is safe.

### Inference is slow on CPU

- Try a smaller model such as `yolov8n.pt`.
- Reduce inference size, for example to `640`.
- Process a shorter clip while debugging.
- Use compatible GPU acceleration if available.

### Too many false alerts

- Re-check polygon placement and camera perspective.
- Increase the confirmation-frame count.
- Review the confidence threshold and tracking stability.
- Test different lighting, crowd levels, and camera positions before drawing conclusions.

### Webhook alerts do not arrive

- Verify the endpoint URL and network access.
- Check the receiving service’s expected request format and authentication requirements.
- Do not put private tokens or credentials in source code or commit them to GitHub.

## Limitations and safety

- A bounding-box foot point is only an approximation of where a person contacts the ground; occlusion and unusual poses can make it inaccurate.
- Fixed polygons are camera-specific and may become invalid if the camera moves or zooms.
- Detection quality varies with resolution, distance, crowd density, lighting, blur, and occlusion.
- Yellow-line detection is experimental and can confuse other yellow objects or fail to see a worn/blocked line.
- The prototype does not independently verify train movement, platform context, or whether a person is authorised to enter a region.
- Confirmation frames and cooldowns trade off responsiveness against false alerts.
- Results must be tested against labelled footage and evaluated using metrics such as precision, recall, false alarms, missed events, and processing FPS.

**Safety notice:** This software is for educational and research use. It is not a certified railway signalling, control, or safety system. Never rely on it as the sole means of preventing accidents or dispatching emergency responses.

## Future improvements

- Evaluate on a labelled, diverse railway-platform dataset.
- Add evaluation scripts for precision, recall, F1 score, false-alarm rate, and FPS.
- Improve tracking under occlusion and crowded scenes.
- Add camera calibration and zone-drawing validation.
- Compare foot-point zone entry with line-crossing and pose-based approaches.
- Add configurable event logs and authenticated, retry-safe alert integrations.
- Add automated tests for polygon geometry, frame confirmation, and cooldown behavior.
- Optimise deployment for edge hardware after accuracy has been validated.

## Contributing

Contributions and suggestions are welcome. For a change:

1. Fork the repository.
2. Create a feature branch.
3. Add or update tests for behavior changes.
4. Run `python test_logic.py`.
5. Submit a pull request describing the change and how it was tested.

Please do not include private CCTV footage, personal information, credentials, or datasets you do not have permission to redistribute.

## License

Choose and add a license before publishing this repository. If you want others to reuse and modify the code, a permissive license such as MIT may be suitable, but confirm that it matches your intentions and the licensing of all included assets and dependencies.
