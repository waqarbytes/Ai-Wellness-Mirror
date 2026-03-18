import cv2

class DashboardOverlay:
    def __init__(self):
        # Colors (BGR)
        self.color_bg = (30, 30, 30)
        self.color_text = (240, 240, 240)
        self.color_accent = (200, 100, 0) # Blue accent
        
        self.color_good = (50, 200, 50)
        self.color_warn = (50, 150, 250) # Orange
        self.color_alert = (50, 50, 220) # Red
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, frame, face_bbox, posture_state, fatigue_state, emotion_tuple, wellness_score, fps):
        """
        Draw bounding box, signals, and dashboard onto the frame.
        """
        if frame is None or frame.size == 0:
            return frame

        h, w, _ = frame.shape
        emotion_state, emotion_conf = emotion_tuple if emotion_tuple is not None else ("Unknown", 0.0)
        
        # --- Draw Face Bounding Box ---
        if face_bbox is not None:
            bx, by, bw, bh = face_bbox
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), self.color_accent, 2)
            cv2.putText(frame, "Face Detected", (bx, max(20, by - 10)), self.font, 0.5, self.color_accent, 1)

        # --- Draw Dashboard Panel ---
        # Panel Background
        panel_w = 280
        panel_h = 200
        margin = 20
        px1, py1 = w - panel_w - margin, margin
        px2, py2 = w - margin, margin + panel_h
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), self.color_bg, -1)
        # Add slight alpha transparency
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (px1, py1), (px2, py2), self.color_accent, 2)

        # Dashboard Text items
        texts = [
            ("AI Wellness Mirror", self.color_accent, 0.6, 2),
            (f"Emotion: {emotion_state} ({emotion_conf:.0f}%)", self.color_text, 0.5, 1),
            (f"Posture: {posture_state}", self._get_posture_color(posture_state), 0.5, 1),
            (f"Fatigue: {fatigue_state}", self._get_fatigue_color(fatigue_state), 0.5, 1),
            (f"Wellness: {wellness_score:.2f}", self._get_wellness_color(wellness_score), 0.5, 1),
            (f"FPS: {fps:.1f}", self.color_text, 0.5, 1)
        ]

        text_y = py1 + 30
        for text, color, scale, thick in texts:
            cv2.putText(frame, text, (px1 + 15, text_y), self.font, scale, color, thick)
            text_y += 25
            
        # Draw Privacy Note at Bottom
        cv2.putText(frame, "Local Processing - No Data Stored", (15, h - 20), self.font, 0.5, self.color_good, 1)
        
        return frame
        
    def _get_posture_color(self, state):
        if state == "Good": return self.color_good
        return self.color_alert

    def _get_fatigue_color(self, state):
        if state == "Normal": return self.color_good
        if state == "Yawning": return self.color_warn
        return self.color_alert

    def _get_wellness_color(self, score):
        if score > 0.8: return self.color_good
        if score > 0.5: return self.color_warn
        return self.color_alert
