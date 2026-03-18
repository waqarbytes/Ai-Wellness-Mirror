from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ModuleNotFoundError:
    mp = None

class FaceDetector:
    def __init__(self, model_path='models/blaze_face_short_range.tflite', min_detection_confidence=0.6, smoothing_factor=0.7):
        """
        Initialize the MediaPipe Face Detection module using Tasks API.
        smoothing_factor (alpha): 1.0 means no smoothing, 0.0 means no update (too much smoothing).
        """
        if mp is None:
            raise ModuleNotFoundError(
                "mediapipe is required for FaceDetector. Install it in the active Python environment."
            )

        resolved_model_path = self._resolve_model_path(model_path)
        if not resolved_model_path.is_file():
            raise FileNotFoundError(f"Face detector model not found: {resolved_model_path}")

        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(resolved_model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence
        )
        self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        self.smoothing_factor = float(np.clip(smoothing_factor, 0.0, 1.0))
        self.prev_bbox = None

    @staticmethod
    def _resolve_model_path(model_path):
        candidate = Path(model_path)
        if candidate.is_file():
            return candidate
        return Path(__file__).resolve().parents[1] / model_path

    def detect(self, frame):
        """
        Detect faces in the frame.
        Returns the bounding box (x, y, w, h) of the first detected face, or None if no face found.
        """
        if frame is None or frame.size == 0:
            self.prev_bbox = None
            return None

        # Convert the BGR image to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = self.detector.detect(mp_image)

        if not results.detections:
            self.prev_bbox = None
            return None

        # Take the highest confidence detection (often the first if only 1 face is active)
        detection = results.detections[0]
        bboxC = detection.bounding_box
        
        ih, iw, _ = frame.shape
        x = int(bboxC.origin_x)
        y = int(bboxC.origin_y)
        w = int(bboxC.width)
        h = int(bboxC.height)
        
        # Clamp to image dimensions
        x = max(0, x)
        y = max(0, y)
        w = min(iw - x, w)
        h = min(ih - y, h)

        curr_bbox = np.array([x, y, w, h], dtype=np.float32)

        if self.prev_bbox is not None:
            # Exponential moving average for smoothing
            smooth_bbox = self.smoothing_factor * curr_bbox + (1 - self.smoothing_factor) * self.prev_bbox
            self.prev_bbox = smooth_bbox
        else:
            self.prev_bbox = curr_bbox
            smooth_bbox = curr_bbox

        return tuple(smooth_bbox.astype(int))

    def close(self):
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
