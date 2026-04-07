"""
modules/emotion.py
==================
Emotion classification using a custom CNN trained on FER2013.
Falls back to DeepFace if the trained model file is not found.
"""

import os
import collections
from pathlib import Path

import numpy as np
import cv2

# Suppress TF logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ── Label map (matches FER2013 class indices) ────────────────────────────────
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMG_SIZE       = 48

# Path to the model produced by train_emotion_model.py
_MODULE_DIR  = Path(__file__).resolve().parent
_MODEL_PATH  = _MODULE_DIR.parent / "models" / "emotion_model.keras"


class EmotionClassifier:
    def __init__(self, history_len: int = 10):
        self.history_len        = history_len
        self.emotion_history    = collections.deque(maxlen=history_len)
        self.confidence_history = collections.deque(maxlen=history_len)

        self._model    = None   # custom Keras model
        self._deepface = None   # fallback

        self._load_model()

    # ── Model loading ────────────────────────────────────────────────────────
    def _load_model(self):
        if _MODEL_PATH.exists():
            try:
                import tensorflow as tf          # noqa: F401
                from tensorflow import keras
                self._model = keras.models.load_model(str(_MODEL_PATH))
                print(f"[EmotionClassifier] Loaded custom model: {_MODEL_PATH}")
                return
            except Exception as exc:
                print(f"[EmotionClassifier] Could not load custom model ({exc}). Falling back to DeepFace.")

        # Fallback: DeepFace
        try:
            from deepface import DeepFace as _DF
            self._deepface = _DF
            print("[EmotionClassifier] Using DeepFace (run train_emotion_model.py to use your own model).")
        except ModuleNotFoundError:
            print("[EmotionClassifier] WARNING: neither custom model nor DeepFace available.")

    # ── Pre-processing ───────────────────────────────────────────────────────
    @staticmethod
    def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
        """Resize & normalise face crop for the custom CNN."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        tensor = resized.astype(np.float32) / 255.0
        return tensor.reshape(1, IMG_SIZE, IMG_SIZE, 1)   # (1, 48, 48, 1)

    # ── Inference ────────────────────────────────────────────────────────────
    def _predict_custom(self, face_bgr: np.ndarray):
        tensor = self._preprocess(face_bgr)
        probs  = self._model.predict(tensor, verbose=0)[0]   # shape (7,)
        idx    = int(np.argmax(probs))
        return EMOTION_LABELS[idx], float(probs[idx] * 100)

    def _predict_deepface(self, frame: np.ndarray, bbox):
        x, y, w, h = (int(v) for v in bbox)
        ih, iw, _  = frame.shape
        x, y = max(0, x), max(0, y)
        w, h = min(iw - x, w), min(ih - y, h)
        face_roi = frame[y:y + h, x:x + w]
        if face_roi.size == 0 or w < 30 or h < 30:
            return "Unknown", 0.0

        analysis = self._deepface.analyze(
            img_path=face_roi,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(analysis, list):
            analysis = analysis[0]
        dominant = analysis["dominant_emotion"]
        conf     = analysis["emotion"][dominant]
        return dominant.capitalize(), float(conf)

    # ── Public API ───────────────────────────────────────────────────────────
    def evaluate(self, frame: np.ndarray, bbox):
        """
        Evaluate emotion on the face region defined by bbox (x, y, w, h).
        Returns (emotion_label: str, confidence_pct: float).
        """
        if frame is None or frame.size == 0 or bbox is None:
            return "Unknown", 0.0

        try:
            if self._model is not None:
                # ── Fast path: custom CNN ──────────────────────────────────
                x, y, w, h = (int(v) for v in bbox)
                ih, iw, _  = frame.shape
                x, y = max(0, x), max(0, y)
                w, h = min(iw - x, w), min(ih - y, h)
                face_roi = frame[y:y + h, x:x + w]
                if face_roi.size == 0 or w < 30 or h < 30:
                    return "Unknown", 0.0
                emotion, confidence = self._predict_custom(face_roi)

            elif self._deepface is not None:
                # ── Fallback: DeepFace ─────────────────────────────────────
                emotion, confidence = self._predict_deepface(frame, bbox)

            else:
                return "Unknown", 0.0

        except Exception:
            return "Unknown", 0.0

        # ── Temporal smoothing ────────────────────────────────────────────
        self.emotion_history.append(emotion)
        self.confidence_history.append(confidence)

        smoothed_emotion     = max(set(self.emotion_history),
                                   key=self.emotion_history.count)
        smoothed_confidence  = float(np.mean(self.confidence_history))

        return smoothed_emotion, smoothed_confidence
