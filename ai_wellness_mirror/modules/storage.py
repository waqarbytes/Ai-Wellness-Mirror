import csv
from datetime import datetime
from pathlib import Path


class DataLogger:
    def __init__(self, username):
        """
        Initializes the logger and creates a CSV file for the session based on the username.
        """
        cleaned_username = str(username).strip().replace(" ", "_")
        self.username = cleaned_username or "Guest"
        
        # Ensure data dir exists
        self.data_dir = Path(__file__).resolve().parents[1] / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Make a personalized file for the user
        filename = f"{self.username}_wellness_log.csv"
        self.filepath = self.data_dir / filename
        
        # Write headers if file doesn't exist
        file_exists = self.filepath.is_file()
        
        self.file = self.filepath.open(mode="a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        
        if not file_exists:
            self.writer.writerow([
                "timestamp", 
                "username", 
                "posture_state", 
                "fatigue_state", 
                "emotion_state", 
                "emotion_confidence", 
                "wellness_score",
                "photo_path"
            ])
            self.file.flush()
            
    def log(self, posture_state, fatigue_state, emotion_tuple, wellness_score, photo_path=""):
        """
        Append a frame's metrics to the CSV file.
        """
        emotion_state, emotion_conf = emotion_tuple if emotion_tuple is not None else ("Unknown", 0.0)
        now = datetime.now().isoformat()
        
        self.writer.writerow([
            now,
            self.username,
            posture_state,
            fatigue_state,
            emotion_state,
            f"{emotion_conf:.2f}",
            f"{float(wellness_score):.2f}",
            photo_path
        ])
        self.file.flush()
        
    def close(self):
        """
        Closes the CSV file pointer.
        """
        if hasattr(self, "file") and self.file:
            self.file.flush()
            self.file.close()
