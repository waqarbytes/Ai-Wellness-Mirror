import numpy as np
import collections

try:
    from deepface import DeepFace
except ModuleNotFoundError:
    DeepFace = None

class EmotionClassifier:
    def __init__(self, history_len=10):
        # We enforce a small history queue for temporal smoothing of emotions
        self.history_len = history_len
        self.emotion_history = collections.deque(maxlen=history_len)
        self.confidence_history = collections.deque(maxlen=history_len)

    def evaluate(self, frame, bbox):
        """
        Evaluate emotion using cropped face ROI from bounding box.
        bbox format: (x, y, w, h)
        """
        if DeepFace is None or bbox is None or frame is None or frame.size == 0:
            return "Unknown", 0.0

        x, y, w, h = (int(value) for value in bbox)
        
        # Ensure coordinates are within image bounds
        ih, iw, _ = frame.shape
        x, y = max(0, x), max(0, y)
        w, h = min(iw-x, w), min(ih-y, h)
        
        # Crop the face
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0 or w < 30 or h < 30:
            return "Unknown", 0.0

        try:
            # enforce_detection=False so we don't double-detect faces, 
            # we just run the Emotion model (FER2013 underneath) on the crop.
            analysis = DeepFace.analyze(
                img_path=face_roi, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True
            )
            
            # DeepFace.analyze might return a list if multiple faces are found (though unlikely on a tight crop)
            if isinstance(analysis, list):
                analysis = analysis[0]
                
            dominant_emotion = analysis['dominant_emotion']
            emotion_probs = analysis['emotion']
            confidence_pct = emotion_probs[dominant_emotion]
            
            # Track history for temporal smoothing
            self.emotion_history.append(dominant_emotion)
            self.confidence_history.append(confidence_pct)
            
            # Temporal smoothing: Most common emotion in last N frames
            smoothed_emotion = max(set(self.emotion_history), key=self.emotion_history.count)
            smoothed_confidence = np.mean(self.confidence_history)
            
            return smoothed_emotion.capitalize(), smoothed_confidence
            
        except Exception:
            # In case inference fails
            return "Unknown", 0.0
