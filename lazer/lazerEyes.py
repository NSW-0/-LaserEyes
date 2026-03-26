import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


class EyeTracker:
    LEFT_EYE_TOP     = 159
    LEFT_EYE_BOTTOM  = 145
    LEFT_EYE_LEFT    = 33
    LEFT_EYE_RIGHT   = 133
    LEFT_IRIS        = 468

    RIGHT_EYE_TOP    = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_EYE_LEFT   = 362
    RIGHT_EYE_RIGHT  = 263
    RIGHT_IRIS       = 473

    EAR_THRESHOLD = 0.25

    def __init__(self):
        base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector    = vision.FaceLandmarker.create_from_options(options)
        self.start_time  = time.time()
        self.frame_count = 0
        self.last_eyes   = []

    def detect(self, frame):
        # Skip every other frame to save CPU
        self.frame_count += 1
        if self.frame_count % 2 != 0:
            return self.last_eyes

        # Process at half resolution — much faster
        small    = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb      = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        results  = self.detector.detect_for_video(mp_image, timestamp_ms)

        eyes = []
        if not results.face_landmarks:
            self.last_eyes = eyes
            return eyes

        h, w = frame.shape[:2]
        lm   = results.face_landmarks[0]

        for side in ['left', 'right']:
            data = self._process_eye(lm, w, h, side)
            if data:
                eyes.append(data)

        self.last_eyes = eyes
        return eyes

    def _process_eye(self, lm, w, h, side):
        if side == 'left':
            top_idx, bottom_idx = self.LEFT_EYE_TOP,   self.LEFT_EYE_BOTTOM
            left_idx, right_idx = self.LEFT_EYE_LEFT,  self.LEFT_EYE_RIGHT
            iris_idx            = self.LEFT_IRIS
        else:
            top_idx, bottom_idx = self.RIGHT_EYE_TOP,  self.RIGHT_EYE_BOTTOM
            left_idx, right_idx = self.RIGHT_EYE_LEFT, self.RIGHT_EYE_RIGHT
            iris_idx            = self.RIGHT_IRIS

        top    = np.array([lm[top_idx].x    * w, lm[top_idx].y    * h])
        bottom = np.array([lm[bottom_idx].x * w, lm[bottom_idx].y * h])
        left   = np.array([lm[left_idx].x   * w, lm[left_idx].y   * h])
        right  = np.array([lm[right_idx].x  * w, lm[right_idx].y  * h])
        iris   = np.array([lm[iris_idx].x   * w, lm[iris_idx].y   * h])

        eye_height = np.linalg.norm(top - bottom)
        eye_width  = np.linalg.norm(right - left)

        if eye_width < 1e-6:
            return None

        # Eye Aspect Ratio check — closed eye = no laser
        ear = eye_height / eye_width
        if ear < self.EAR_THRESHOLD:
            return None

        # Gaze direction from iris offset relative to eye center
        eye_center     = (left + right) / 2.0
        iris_offset    = iris - eye_center
        relative_offset = iris_offset / (eye_width * 0.5)

        # Amplify the offset so small eye movements = big laser movement
        gaze_dir = np.array([relative_offset[0] * 3.0, relative_offset[1] * 3.0])
        length   = np.linalg.norm(gaze_dir)
        gaze_dir = gaze_dir / length if length > 1e-6 else np.array([1.0, 0.0])

        return (iris.astype(int), gaze_dir, eye_width)


class RedLaserRenderer:
    CORE_COLOR = (255, 255, 255)  # white core
    BEAM_COLOR = (0,   0,   255)  # red beam
    GLOW_COLOR = (0,   0,   180)  # darker red glow

    def draw(self, frame, eyes):
        for origin, gaze_dir, eye_width in eyes:
            self._draw_laser(frame, origin, gaze_dir, eye_width)

    def _draw_laser(self, frame, origin, gaze_dir, eye_width):
        h, w   = frame.shape[:2]
        beam_w = max(1, int(eye_width * 0.04))

        # Shoot to screen edge in gaze direction
        end = self._ray_to_edge(origin, gaze_dir, w, h)

        # Outer glow
        glow_layer = frame.copy()
        cv2.line(glow_layer, tuple(origin), end, self.GLOW_COLOR, beam_w * 6, cv2.LINE_AA)
        cv2.addWeighted(glow_layer, 0.3, frame, 0.7, 0, frame)

        # Middle beam
        cv2.line(frame, tuple(origin), end, self.BEAM_COLOR, beam_w * 3, cv2.LINE_AA)

        # Bright white core
        cv2.line(frame, tuple(origin), end, self.CORE_COLOR, beam_w, cv2.LINE_AA)

        # Flash dot at origin
        cv2.circle(frame, tuple(origin), beam_w * 3, self.CORE_COLOR, -1,     cv2.LINE_AA)
        cv2.circle(frame, tuple(origin), beam_w * 5, self.BEAM_COLOR, beam_w, cv2.LINE_AA)

    def _ray_to_edge(self, origin, direction, w, h):
        """Find where the gaze ray hits the screen edge."""
        ox, oy = float(origin[0]), float(origin[1])
        dx, dy = direction
        candidates = []
        if abs(dx) > 1e-6:
            candidates.append((w - ox) / dx if dx > 0 else -ox / dx)
        if abs(dy) > 1e-6:
            candidates.append((h - oy) / dy if dy > 0 else -oy / dy)
        positives = [t for t in candidates if t > 0]
        t = min(positives) if positives else 600.0
        end_x = int(ox + dx * t)
        end_y = int(oy + dy * t)
        return (end_x, end_y)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker   = EyeTracker()
    renderer  = RedLaserRenderer()
    prev_time = time.time()

    print("👁️  Eye Laser running — open your eyes wide and look around!")
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        eyes  = tracker.detect(frame)

        if eyes:
            renderer.draw(frame, eyes)

        now  = time.time()
        fps  = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

        cv2.imshow("Eye Lasers", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()