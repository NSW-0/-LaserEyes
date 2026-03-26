# 👁️ LaserEyes

A real-time augmented reality app that shoots **red laser beams from your eyes** using gaze direction detected via MediaPipe Face Mesh and OpenCV.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-red)

---

## ✨ Features

- 🔴 Red laser beam with glowing core shoots from each eye
- 👁️ **Gaze tracking** — laser follows where you're actually looking
- 😑 **Blink to stop** — lasers only fire when your eyes are open
- ⚡ Optimized for low CPU/memory usage
- 🎯 Supports both eyes simultaneously

---

## 🛠️ Requirements

- Python 3.10+
- Webcam

### Dependencies

```bash
pip install opencv-python mediapipe numpy
```

---

## 🚀 Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/LaserEyes.git
cd LaserEyes
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install opencv-python mediapipe numpy
```

**4. Download the face landmark model**
```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task', 'face_landmarker.task')"
```

**5. Run the app**
```bash
python laser_eyes.py
```

Press **Q** to quit.

---

## 📁 Project Structure

```
LaserEyes/
├── laser_eyes.py          # Main application
├── face_landmarker.task   # MediaPipe model file (downloaded separately)
└── README.md
```

---

## ⚙️ How It Works

| Component | Description |
|---|---|
| `EyeTracker` | Uses MediaPipe Face Landmarker to detect iris position and eye landmarks |
| `_process_eye()` | Calculates gaze direction from iris offset relative to eye center |
| `EAR check` | Eye Aspect Ratio — lasers only fire when eyes are sufficiently open |
| `RedLaserRenderer` | Draws a glowing red beam from the iris to the screen edge in gaze direction |
| `_ray_to_edge()` | Computes where the gaze ray intersects the screen boundary |

---

## 🎨 Customization

**Change laser color** inside `RedLaserRenderer` (BGR format):
```python
CORE_COLOR = (255, 255, 255)  # white core
BEAM_COLOR = (0,   0,   255)  # red beam   → try (255, 0, 0) for blue
GLOW_COLOR = (0,   0,   180)  # darker glow
```

**Adjust gaze sensitivity** in `_process_eye()`:
```python
gaze_dir = np.array([relative_offset[0] * 3.0, relative_offset[1] * 3.0])
#                                          ↑ increase for more sensitive gaze tracking
```

**Adjust eye open strictness** — raise to require wider open eyes, lower if lasers don't fire enough:
```python
EAR_THRESHOLD = 0.25  # try 0.20 (easier) or 0.30 (stricter)
```

---

## ⚡ Performance Optimizations

This project is built with performance in mind:

- **640x480 resolution** — camera capped to reduce processing load
- **Half-res detection** — MediaPipe runs on a 50% scaled frame, results mapped back to full size
- **Frame skipping** — face detection runs every 2nd frame, last result reused in between

---

## 🔧 Troubleshooting

**Lasers fire even when eyes are closed**
Raise the `EAR_THRESHOLD` value in `EyeTracker`:
```python
EAR_THRESHOLD = 0.30  # stricter open eye requirement
```

**Lasers don't fire at all**
Lower the `EAR_THRESHOLD` value:
```python
EAR_THRESHOLD = 0.18  # easier to trigger
```

**Gaze direction feels off**
Make sure your face is well lit and centered in the frame. Gaze tracking works best when your face is roughly facing the camera.

**`face_landmarker.task` not found**
Re-run the model download command from Step 4 above.

---

## 🤝 Related Projects

Also check out **[MagicHands](https://github.com/your-username/MagicHands)** — hold an open palm toward the camera to summon a spinning magic circle around your hand!

---

## 📄 License

This project is licensed under the MIT License.
