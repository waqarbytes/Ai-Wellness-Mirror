import cv2
import time

from modules.face_detection import FaceDetector
from modules.landmarks import LandmarkExtractor
from modules.head_pose import HeadPoseEstimator
from modules.fatigue import FatigueEvaluator
from modules.emotion import EmotionClassifier
from modules.fusion import SignalFusion
from modules.overlay import DashboardOverlay
from modules.storage import DataLogger

def main():
    print("Initializing AI Wellness Mirror...")
    
    # Prompt User For Name
    username = input("\nWelcome to AI Wellness Mirror!\nPlease enter your name to start your session: ")
    if not username.strip():
        username = "Guest"
    print(f"\nStarting session for {username}. Look into the webcam...\n")
    
    # Initialize all modules
    detector = FaceDetector(smoothing_factor=0.6)
    landmarks = LandmarkExtractor()
    head_pose = HeadPoseEstimator()
    fatigue = FatigueEvaluator()
    emotion = EmotionClassifier()
    fusion = SignalFusion()
    overlay = DashboardOverlay()
    logger = DataLogger(username)

    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Optional: Lower resolution to ensure high FPS on slower CPUs
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    import os
    print("System ready. Press 'q' to quit.")

    prev_time = time.time()
    photo_captured = False
    
    images_dir = os.path.join("data", "images")
    os.makedirs(images_dir, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # FPS Calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # 1. Face Detection
        face_bbox = detector.detect(frame)
        
        posture_state = "Unknown"
        fatigue_state = "Unknown"
        emotion_state, emotion_conf = "Unknown", 0.0
        wellness_score = 0.5
        
        if face_bbox is not None:
            # 2. Landmark Extraction
            landmarks_2d = landmarks.extract(frame)
            
            if landmarks_2d is not None:
                # 3. Head Pose & Posture
                _, _, _, posture_state = head_pose.evaluate(landmarks_2d, frame.shape)
                
                # 4. Fatigue Analysis
                fatigue_state, f_score, _, _ = fatigue.evaluate(landmarks_2d)
                
                # 5. Emotion Classification (Runs on bounding box crop)
                # Note: DeepFace runs in same thread. If latency is high, this could be threaded.
                emotion_state, emotion_conf = emotion.evaluate(frame, face_bbox)
                
                # 6. Signal Fusion
                wellness_score = fusion.compute_wellness(posture_state, f_score, emotion_state)
                
                # Capture photo on first successful face detection
                photo_path = ""
                if not photo_captured:
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    photo_filename = f"{username.strip().replace(' ', '_')}_{timestamp_str}.jpg"
                    photo_path_full = os.path.join(images_dir, photo_filename)
                    cv2.imwrite(photo_path_full, frame)
                    photo_path = photo_path_full
                    photo_captured = True
                    print(f"\n[INFO] Photo captured and saved to: {photo_path_full}\n")

                # 8. Log Data
                logger.log(posture_state, fatigue_state, (emotion_state, emotion_conf), wellness_score, photo_path)
                
        # 7. Render Overlay
        frame = overlay.render(
            frame, 
            face_bbox, 
            posture_state, 
            fatigue_state, 
            (emotion_state, emotion_conf), 
            wellness_score, 
            fps
        )
        
        cv2.imshow('AI Wellness Mirror', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    landmarks.close()
    logger.close()
    print(f"\nSession saved to data/{username.strip().replace(' ', '_')}_wellness_log.csv!")

if __name__ == "__main__":
    main()
