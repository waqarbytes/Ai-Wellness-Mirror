import cv2
import numpy as np

class HeadPoseEstimator:
    def __init__(self, tilt_threshold=20.0, history_len=10):
        # MediaPipe standard landmark indices for PnP
        self.indices = [
            1,      # Nose tip
            152,    # Chin
            33,     # Left eye (subject's right) outer corner
            263,    # Right eye (subject's left) outer corner
            61,     # Left mouth corner
            291     # Right mouth corner
        ]
        
        # Generic 3D model points for a human face
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner
            (225.0, 170.0, -135.0),      # Right eye corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        self.tilt_threshold = tilt_threshold
        self.history_len = history_len
        self.pitch_history = []
        
    def evaluate(self, landmarks_2d, frame_shape):
        """
        Evaluate head pose and return pitch, yaw, roll, and posture status string.
        landmarks_2d: full list of 478 landmarks (x, y)
        frame_shape: (height, width, channels)
        """
        if not landmarks_2d or len(landmarks_2d) < 292:
            return 0, 0, 0, "Unknown"

        image_pts = np.array([landmarks_2d[i] for i in self.indices], dtype=np.float64)
        
        if len(frame_shape) < 2:
            return 0, 0, 0, "Unknown"

        h, w = frame_shape[:2]
        # Camera internals (focal length = width for estimation)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64
        )
        
        dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0, 0, 0, "Tracking Failed"
            
        # Get rotation matrix
        rmat, _ = cv2.Rodrigues(rotation_vector)
        
        # Extract Euler angles (Pitch, Yaw, Roll)
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rmat[2, 1], rmat[2, 2])
            y = np.arctan2(-rmat[2, 0], sy)
            z = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            x = np.arctan2(-rmat[1, 2], rmat[1, 1])
            y = np.arctan2(-rmat[2, 0], sy)
            z = 0
            
        pitch = np.degrees(x)
        yaw = np.degrees(y)
        roll = np.degrees(z)

        # Update history
        self.pitch_history.append(pitch)
        if len(self.pitch_history) > self.history_len:
            self.pitch_history.pop(0)
            
        # Determine posture (average absolute pitch)
        avg_pitch = np.mean(self.pitch_history)
        if abs(avg_pitch) > self.tilt_threshold or abs(yaw) > self.tilt_threshold * 1.5:
            posture = "Slouched / Tilted"
        else:
            posture = "Good"

        return pitch, yaw, roll, posture
