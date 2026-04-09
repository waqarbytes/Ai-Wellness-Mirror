class SignalFusion:
    def __init__(self):
        # Wellness weights  (must sum to 1.0)
        self.w_posture = 0.35
        self.w_fatigue = 0.25
        self.w_emotion = 0.25
        self.w_voice   = 0.15   # vocal state from RAVDESS-trained SVM

    def compute_wellness(self, posture_state, fatigue_score, emotion_state, vocal_state="Unknown"):
        """
        Synthesize multiple signals into a simple wellness score (0 to 1).
        Does not derive psychological conclusions.

        Args:
            posture_state: str from HeadPoseEstimator
            fatigue_score: float 0.0 (normal) – 1.0 (highly fatigued)
            emotion_state: str from EmotionClassifier
            vocal_state:   str from VoiceAnalyzer ('calm'|'stressed'|'fatigued'|'Unknown')
        """
        posture_state = str(posture_state)
        emotion_state = str(emotion_state)
        vocal_state   = str(vocal_state)
        fatigue_score = max(0.0, min(1.0, float(fatigue_score)))

        # Posture scoring
        if posture_state == "Good":
            post_score = 1.0
        elif posture_state in {"Unknown", "Tracking Failed"}:
            post_score = 0.5
        else:
            post_score = 0.0  # Slouched / Tilted

        # Fatigue scoring (fatigue_score is 0.0 = Normal, 1.0 = Highly Fatigued)
        fat_score = 1.0 - fatigue_score

        # Emotion scoring
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

        # Vocal state scoring
        v = vocal_state.lower()
        if v == "calm":
            voc_score = 1.0
        elif v == "fatigued":
            voc_score = 0.4
        elif v == "stressed":
            voc_score = 0.2
        else:  # Unknown / not yet classified
            voc_score = 0.5

        # Final weighted score
        wellness_idx = (
            self.w_posture * post_score
            + self.w_fatigue * fat_score
            + self.w_emotion * emo_score
            + self.w_voice   * voc_score
        )

        return wellness_idx
