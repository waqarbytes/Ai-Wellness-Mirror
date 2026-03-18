class SignalFusion:
    def __init__(self):
        # Wellness weights
        self.w_posture = 0.4
        self.w_fatigue = 0.3
        self.w_emotion = 0.3

    def compute_wellness(self, posture_state, fatigue_score, emotion_state):
        """
        Synthesize multiple signals into a simple wellness score (0 to 1).
        Does not derive psychological conclusions.
        """
        posture_state = str(posture_state)
        emotion_state = str(emotion_state)
        fatigue_score = max(0.0, min(1.0, float(fatigue_score)))

        # Posture scoring
        if posture_state == "Good":
            post_score = 1.0
        elif posture_state in {"Unknown", "Tracking Failed"}:
            post_score = 0.5
        else:
            post_score = 0.0  # Slouched / Tilted
            
        # Fatigue scoring (fatigue_score is 0.0 = Normal, 1.0 = Highly Fatigued)
        # We invert it so 1.0 = highly awake/well
        fat_score = 1.0 - fatigue_score
        
        # Emotion scoring (neutral interpretation mapping)
        e = emotion_state.lower()
        if e in ["happy", "neutral"]:
            emo_score = 1.0
        elif e == "surprise":
            emo_score = 0.8
        elif e == "sad":
            emo_score = 0.5
        elif e in ["fear", "angry", "disgust"]:
            emo_score = 0.3
        else:
            emo_score = 0.5
            
        # Final weighted score
        wellness_idx = (
            self.w_posture * post_score + 
            self.w_fatigue * fat_score + 
            self.w_emotion * emo_score
        )
        
        return wellness_idx
