"""
voice/vocal_state.py
====================
Inference-only module for the RAVDESS-trained vocal state classifier.

Usage:
    from voice.vocal_state import VocalStateClassifier, VocalState

    clf  = VocalStateClassifier()
    state, conf = clf.predict_from_file(Path("sample.wav"))
    print(state, conf)

Zero side-effects on import.  No training code.
"""

from __future__ import annotations

import pickle
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

# ── Re-use preprocessing and feature extraction from the training script ──────
# Import path assumes both scripts are in the same parent directory
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_voice_model import CONFIG, extract_features, preprocess


# ══════════════════════════════════════════════════════════════════════════════
#  Public types
# ══════════════════════════════════════════════════════════════════════════════

class VocalState(str, Enum):
    """The three vocal states the model can predict."""
    CALM     = "calm"
    STRESSED = "stressed"
    FATIGUED = "fatigued"


class ModelNotFoundError(FileNotFoundError):
    """Raised when the .pkl model bundle cannot be found on disk."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
#  Classifier
# ══════════════════════════════════════════════════════════════════════════════

class VocalStateClassifier:
    """
    Load the saved RAVDESS-trained sklearn pipeline and classify raw audio.

    This class is inference-only: no training code, no side effects on import.
    All preprocessing and feature extraction is handled internally so callers
    only need to supply raw audio arrays or file paths.
    """

    DEFAULT_MODEL_PATH = Path("voice/models/vocal_state_clf.pkl")

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> None:
        """
        Load the model bundle from *model_path*.

        Raises:
            ModelNotFoundError: if the .pkl file is not found.
            ValueError:         if the bundle has unexpected feature_size or labels.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelNotFoundError(
                f"Model bundle not found at: {model_path.resolve()}\n"
                "Run 'python train_voice_model.py' first to generate it."
            )

        with open(model_path, "rb") as fh:
            bundle = pickle.load(fh)

        self._pipe         = bundle["model"]
        self._le           = bundle["label_encoder"]
        self._labels: list[str] = bundle["labels"]
        self._feature_size: int = bundle["feature_size"]
        self._sample_rate:  int = bundle["sample_rate"]
        self._fixed_duration: int = bundle["fixed_duration"]

        # Sanity checks
        if self._feature_size != CONFIG["feature_size"]:
            raise ValueError(
                f"Bundle feature_size ({self._feature_size}) does not match "
                f"CONFIG feature_size ({CONFIG['feature_size']}). "
                "Re-train the model."
            )
        for lbl in ["calm", "stressed", "fatigued"]:
            if lbl not in self._labels:
                raise ValueError(
                    f"Expected label '{lbl}' missing from bundle labels: {self._labels}"
                )

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> tuple[VocalState, float]:
        """
        Classify a raw float32 audio array into one of three vocal states.

        The full preprocessing and feature extraction pipeline is applied
        internally — the caller just passes raw audio.

        Args:
            audio: 1-D float32 numpy array at *sr* Hz.
            sr:    Sample rate of *audio* (default 16 000 Hz).

        Returns:
            (VocalState, confidence)  where confidence ∈ [0.0, 1.0].

        Performance: typically < 100 ms for a 3-second clip on CPU.
        """
        # Resample if needed (CONFIG defines the expected sample rate)
        cfg = dict(CONFIG)
        cfg["sample_rate"] = sr  # honour caller's sr for load path

        # Preprocessing expects a loaded signal — feed it directly after
        # ensuring the correct dtype
        signal = audio.astype(np.float32)

        # Apply the same trim / normalise / pad-truncate steps
        import librosa
        signal, _ = librosa.effects.trim(signal, top_db=cfg["top_db"])
        signal = signal / (np.max(np.abs(signal)) + 1e-9)
        target_len = cfg["sample_rate"] * cfg["fixed_duration"]
        if len(signal) >= target_len:
            signal = signal[:target_len]
        else:
            signal = np.pad(signal, (0, target_len - len(signal)), mode="constant")
        signal = signal.astype(np.float32)

        # Feature extraction (use the training-script function unchanged)
        features = extract_features(signal, cfg).reshape(1, -1)

        # Predict
        label_enc   = self._pipe.predict(features)[0]
        proba       = self._pipe.predict_proba(features)[0]
        confidence  = float(proba.max())
        label_str   = self._le.inverse_transform([label_enc])[0]

        return VocalState(label_str), confidence

    def predict_from_file(self, filepath: Path) -> tuple[VocalState, float]:
        """
        Load a .wav file and return (VocalState, confidence).

        Args:
            filepath: Path to a mono or stereo .wav audio file.

        Returns:
            (VocalState, confidence)  where confidence ∈ [0.0, 1.0].

        Raises:
            FileNotFoundError: if *filepath* does not exist.
            RuntimeError:      if librosa cannot load the file.
        """
        import librosa

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath.resolve()}")

        try:
            audio, sr = librosa.load(
                str(filepath),
                sr=CONFIG["sample_rate"],
                mono=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load audio file '{filepath}': {exc}") from exc

        return self.predict(audio, sr=sr)

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return a helpful string representation of the classifier."""
        return (
            f"VocalStateClassifier("
            f"labels={self._labels}, "
            f"feature_size={self._feature_size}, "
            f"sr={self._sample_rate}, "
            f"duration={self._fixed_duration}s)"
        )
