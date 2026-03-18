import numpy as np


class FatigueEvaluator:
    def __init__(self, ear_threshold=0.21, ear_frames=10, mar_threshold=0.5, mar_frames=15):
        # MediaPipe eye landmarks
        # Right Eye (image left)
        self.right_eye_indices = [33, 160, 158, 133, 153, 144]
        # Left Eye (image right)
        self.left_eye_indices = [362, 385, 387, 263, 373, 380]
        
        # MediaPipe mouth landmarks
        # Inner lips points are generally better for MAR
        self.mouth_indices_inner = {
            'left': 78,
            'right': 308,
            'top1': 13,
            'bottom1': 14,
            'top2': 81,
            'bottom2': 311,
            'top3': 82,
            'bottom3': 312
        }

        self.ear_threshold = ear_threshold
        self.ear_frames = ear_frames
        self.mar_threshold = mar_threshold
        self.mar_frames = mar_frames
        
        self.blink_counter = 0
        self.yawn_counter = 0
        
        self.total_blinks = 0
        self.total_yawns = 0

    @staticmethod
    def _euclidean(point_a, point_b):
        return float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))

    def calculate_ear(self, eye_pts):
        # Compute the euclidean distances between the two sets of vertical eye landmarks
        A = self._euclidean(eye_pts[1], eye_pts[5])
        B = self._euclidean(eye_pts[2], eye_pts[4])
        # Horizontal distance
        C = self._euclidean(eye_pts[0], eye_pts[3])
        # Calculate eye aspect ratio
        if C == 0:
            return 0.0
        ear = (A + B) / (2.0 * C)
        return ear

    def calculate_mar(self, landmarks_2d):
        p_left = landmarks_2d[self.mouth_indices_inner['left']]
        p_right = landmarks_2d[self.mouth_indices_inner['right']]
        p_top = landmarks_2d[self.mouth_indices_inner['top1']]
        p_bottom = landmarks_2d[self.mouth_indices_inner['bottom1']]

        # Horizontal distance
        C = self._euclidean(p_left, p_right)
        # Vertical distance
        V = self._euclidean(p_top, p_bottom)
        
        if C == 0:
            return 0.0
        return V / C

    def evaluate(self, landmarks_2d):
        """
        Evaluate fatigue metrics based on landmarks.
        Returns: fatigue_state (string), score (float), current_ear, current_mar
        """
        min_required_landmarks = max(max(self.left_eye_indices), max(self.mouth_indices_inner.values()))
        if not landmarks_2d or len(landmarks_2d) <= min_required_landmarks:
            return "Unknown", 0.0, 0.0, 0.0
            
        right_eye = [landmarks_2d[i] for i in self.right_eye_indices]
        left_eye = [landmarks_2d[i] for i in self.left_eye_indices]
        
        ear_right = self.calculate_ear(right_eye)
        ear_left = self.calculate_ear(left_eye)
        
        ear = (ear_right + ear_left) / 2.0
        mar = self.calculate_mar(landmarks_2d)
        
        # Blink/Eye closure logic
        is_closing = False
        if ear < self.ear_threshold:
            self.blink_counter += 1
            if self.blink_counter >= self.ear_frames:
                is_closing = True
        else:
            if self.blink_counter >= 2 and self.blink_counter < self.ear_frames:
                # Registered a proper short blink
                self.total_blinks += 1
            self.blink_counter = 0
            
        # Yawn logic
        is_yawning = False
        if mar > self.mar_threshold:
            self.yawn_counter += 1
            if self.yawn_counter >= self.mar_frames:
                is_yawning = True
        else:
            if self.yawn_counter >= self.mar_frames:
                self.total_yawns += 1 # Finished yawning
            self.yawn_counter = 0

        # Construct Fatigue Score (0.0=Normal to 1.0=Highly Fatigued)
        # For prototype: simple logic based on current frame closure/yawn plus accrued yawns
        score = 0.0
        state = "Normal"
        
        if is_closing:
            score += 0.8
            state = "Drowsing / Sleeping"
        elif is_yawning:
            score += 0.5
            state = "Yawning"
            
        if self.total_yawns > 3 and not is_closing:
            score += 0.4
            state = "Fatigued"
            
        score = min(1.0, score)
        
        return state, score, ear, mar
