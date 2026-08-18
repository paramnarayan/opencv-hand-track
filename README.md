# Hand Tracking Image Reveal

A real-time MediaPipe and OpenCV demo that reveals a fixed image through a
window formed by two hands. The image remains centered while the index fingers
and thumbs control the reveal area.

## Features

- MediaPipe Hand Landmarker with an optional CVZone fallback
- One Euro Filter smoothing using a monotonic clock
- Convex-area validation to reject collapsed or crossed hand shapes
- Clean image reveal with no colored hand-polygon tint
- Distinct colored edge guides for the top, right, bottom, and left sides
- Short dropout holding and implausible-jump rejection for steadier tracking
- Reduced-resolution detector input mapped back to the full camera frame
- Reused render buffers to reduce per-frame memory allocation
- Command-line configuration without editing source code
- Automatic cleanup of cameras, windows, and detector resources

## Requirements

- Python 3.10 or newer
- A webcam
- The bundled `hand_landmarker.task` model, or another compatible model

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the optional CVZone fallback with:

```bash
python -m pip install -e ".[fallback]"
```

## Running

The original entry point remains supported:

```bash
python main.py
```

After an editable installation, either of these also works:

```bash
hand-reveal
python -m hand_reveal
```

Press `Q` or `ESC` to quit.

## Camera discovery

Camera scanning is opt-in so startup does not open every camera automatically:

```bash
python main.py --list-cameras
python main.py --camera 1
```

## Common options

```text
--image PATH              Image to reveal
--model PATH              MediaPipe task model
--camera INDEX            Camera index
--rotate {none,cw,ccw,180}
--mirror {auto,on,off}    Camera-specific mirror correction
--width PIXELS            Requested capture width; 0 uses camera profile
--height PIXELS           Requested capture height; 0 uses camera profile
--min-gap PIXELS          Minimum index-finger separation
--min-area PIXELS         Minimum valid reveal polygon area
--image-size FRACTION     Display size relative to the camera frame
--feather-radius PIXELS   Reveal-edge softness; 0 disables it
--min-cutoff VALUE        One Euro Filter stationary smoothing
--beta VALUE              One Euro Filter motion responsiveness
--inference-size PIXELS   Maximum detector input dimension; 0 uses full size
--dropout-hold FRAMES     Frames to retain the last reliable hand polygon
--max-quad-jump FRACTION  Reject implausible one-frame tracking jumps
--fps VALUE               Requested camera frame rate
```

Run `python main.py --help` for the complete list.

### Examples

Use an iPhone Continuity Camera commonly exposed as camera 1:

```bash
python main.py --camera 1
```

Camera 0 requests 1280x720 and receives mirror correction by default. Camera 1
requests 1920x1080 and keeps the iPhone/virtual-camera orientation unchanged.
The camera may return a lower resolution when its driver does not support the
requested profile. Use `--mirror on`, `--mirror off`, or `--rotate` only when a
specific camera driver needs an override.

Use a custom image:

```bash
python main.py --image path/to/photo.jpg
```

Rotate a sideways camera feed:

```bash
python main.py --rotate cw
```

Image and model paths are resolved independently of the terminal's working
directory. If the requested image cannot be read, the application searches the
same directory for another supported image and finally generates a placeholder.
The image uses 50% of the available frame by default while preserving its
original aspect ratio.

## Project structure

```text
opencv-hand-track/
├── main.py
├── pyproject.toml
├── hand_landmarker.task
├── testimage.jpg
└── src/hand_reveal/
    ├── app.py
    ├── cli.py
    ├── config.py
    ├── detector.py
    ├── filters.py
    ├── geometry.py
    └── renderer.py
```

GitHub Actions verifies packaging, compilation, imports, and all three command
entry points on supported Python versions. Hardware-dependent camera execution
is intentionally not run in CI.

## Troubleshooting

- Use `--list-cameras` if the default camera cannot be opened.
- Increase `--min-area` if accidental narrow shapes reveal the image.
- Reduce `--feather-radius` or `--image-size` on slower hardware.
- Verify that the model path points to a readable Hand Landmarker task file.

## License

MIT. See `LICENSE`.
