from pathlib import Path

import cv2

try:
    import mediapipe as mp
except ModuleNotFoundError:
    mp = None


class LandmarkExtractor:
    def __init__(self, model_path='models/face_landmarker.task', max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize MediaPipe FaceMesh for extracting 478 landmarks using Tasks API.
        """
        if mp is None:
            raise ModuleNotFoundError(
                "mediapipe is required for LandmarkExtractor. Install it in the active Python environment."
            )

        resolved_model_path = self._resolve_model_path(model_path)
        if not resolved_model_path.is_file():
            raise FileNotFoundError(f"Face landmarker model not found: {resolved_model_path}")

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(resolved_model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    @staticmethod
    def _resolve_model_path(model_path):
        candidate = Path(model_path)
        if candidate.is_file():
            return candidate
        return Path(__file__).resolve().parents[1] / model_path

    def extract(self, frame):
        """
        Extract landmarks from the full frame. 
        Returns a list of 2D coordinates (x, y) for the first face found.
        """
        if frame is None or frame.size == 0:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # In VIDEO running mode, we must supply a strictly increasing timestamp
        self.timestamp_ms += 33
        results = self.face_mesh.detect_for_video(mp_image, self.timestamp_ms)

        if not results.face_landmarks:
            return None

        ih, iw, _ = frame.shape
        landmarks_2d = []
        
        # Take the first face
        face_landmarks = results.face_landmarks[0]
        
        for lm in face_landmarks:
            x = int(lm.x * iw)
            y = int(lm.y * ih)
            landmarks_2d.append((x, y))

        return landmarks_2d

    def close(self):
        if hasattr(self, "face_mesh") and self.face_mesh is not None:
            self.face_mesh.close()
