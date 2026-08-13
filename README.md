# 🖐️ Hand Tracking Image Reveal

A real-time hand-tracking demo using **MediaPipe** and **OpenCV** that lets you reveal a fixed image by forming a frame with your two hands. The image is pinned at the center of the screen — your hands act as a reveal window, uncovering the image as you move them over it.

---

## ✨ How It Works

1. Hold both hands up in front of your webcam
2. Form a box/frame shape using your **index fingers** and **thumbs**
3. The area between your hands acts as a **transparent window** that reveals the image beneath
4. The image stays locked at the center of the screen — only your hands move

---

## 📋 Requirements

- Python 3.10+
- A webcam (or iPhone via Continuity Camera on macOS)
- The `hand_landmarker.task` model file (MediaPipe)

### Install dependencies

```bash
pip install opencv-python numpy mediapipe
```

> **Optional:** Install `cvzone` as a fallback detector:
> ```bash
> pip install cvzone
> ```

---

## 🚀 Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/paramnarayan/opencv-hand-track.git
   cd opencv-hand-track
   ```

2. **Add your image**  
   Drop any `.jpg`, `.jpeg`, `.png`, or `.webp` image into the project folder and update `IMAGE_PATH` in `main.py`:
   ```python
   IMAGE_PATH = "your_photo.jpg"
   ```
   If no image is set, the script auto-detects the first image it finds in the folder.

3. **Download the MediaPipe model**  
   Download `hand_landmarker.task` from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models) and place it in the project root.

4. **Run**
   ```bash
   python3 main.py
   ```

Press **Q** or **ESC** to quit.

---

## ⚙️ Configuration

All settings are at the top of `main.py`:

| Variable | Default | Description |
|---|---|---|
| `IMAGE_PATH` | `"testimage.jpg"` | Path to the image to reveal |
| `MODEL_PATH` | `"hand_landmarker.task"` | Path to MediaPipe model |
| `CAMERA_INDEX` | `0` | Camera to use (`0` = built-in, `1` = iPhone) |
| `FLIP_CAMERA` | `True` | Mirror the frame horizontally (set `False` for iPhone) |
| `ROTATE_FRAME` | `None` | Rotate frame if sideways (see iPhone section) |
| `IMG_DISPLAY_FRAC` | `0.35` | Image size as fraction of the smaller screen dimension |
| `MIN_GAP` | `80` | Minimum hand spread (px) before image appears |
| `OEF_MIN_CUTOFF` | `2.0` | Smoothing when hands are still — lower = smoother, slightly laggier |
| `OEF_BETA` | `0.5` | Responsiveness when moving fast — higher = less lag |
| `ORANGE_STRENGTH` | `0.22` | Orange tint intensity inside the hand quad |
| `FEATHER_RADIUS` | `5` | Soft edge feathering radius (px) |

---

## 📱 Using an iPhone as Camera (macOS only)

macOS **Continuity Camera** lets you use your iPhone as a high-quality webcam wirelessly.

**Requirements:**
- iPhone on iOS 16+
- Mac on macOS Ventura (13)+
- Same Apple ID signed in on both devices
- Wi-Fi and Bluetooth enabled on both

**Setup:**
1. Bring your iPhone near your Mac — it appears automatically
2. In `main.py`, set:
   ```python
   CAMERA_INDEX = 1      # iPhone is usually index 1
   FLIP_CAMERA  = False  # iPhone feed is already correctly oriented
   ```
3. If the image appears sideways, add:
   ```python
   ROTATE_FRAME = cv2.ROTATE_90_CLOCKWISE
   # or
   ROTATE_FRAME = cv2.ROTATE_90_COUNTERCLOCKWISE
   ```

Run the script once to see all detected cameras listed in the startup log.

---

## 🧠 Technical Details

- **Detector:** MediaPipe `HandLandmarker` (high-precision task API, VIDEO mode). Falls back to `cvzone` if the `.task` model is missing.
- **Smoothing:** [One Euro Filter](https://cristal.univ-lille.fr/~casiez/1euro/) (Casiez et al. CHI 2012) — adapts its cutoff frequency to motion speed. Near-zero lag when moving fast; jitter-free when still. Tuned via `OEF_MIN_CUTOFF` and `OEF_BETA`.
- **Rendering:** Two-layer compositing — orange tint inside the hand quad, then the fixed image composited through a feathered reveal mask. All buffers are pre-allocated at startup for zero per-frame heap allocation.
- **Image position:** Computed once from frame dimensions at startup. The image never moves or scales during runtime.
- **Camera:** Requests 60 FPS from the capture device for minimum latency.

---

## 📁 Project Structure

```
opencv-hand-track/
├── main.py                # Main application
├── hand_landmarker.task   # MediaPipe hand model (download separately)
├── testimage.jpg          # Demo image (replace with your own)
└── README.md
```

---

## 📄 License

MIT
