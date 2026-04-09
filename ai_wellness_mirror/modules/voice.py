"""
modules/voice.py
================
Thin wrapper around VocalStateClassifier for use inside the AI Wellness Mirror.

Samples audio from the microphone in a background thread, classifies it every
N seconds, and exposes the latest (state, confidence) via a thread-safe property.
Falls back gracefully if the model is not found or microphone is unavailable.
"""

import threading
import time
from pathlib import Path
from typing import Optional
import sys

import numpy as np

# ── Locate the vocal_state module relative to this package ────────────────────
_FACESYNC_ROOT = Path(__file__).resolve().parent.parent.parent  # Desktop/Facesync
if str(_FACESYNC_ROOT) not in sys.path:
    sys.path.insert(0, str(_FACESYNC_ROOT))

try:
    from voice.vocal_state import VocalState, VocalStateClassifier, ModelNotFoundError
    _VOCAL_STATE_AVAILABLE = True
except Exception:
    _VOCAL_STATE_AVAILABLE = False

# Sample duration and rate must match training config
_SAMPLE_RATE   = 16000
_DURATION_SECS = 3
_CHUNK_SAMPLES = _SAMPLE_RATE * _DURATION_SECS   # 48 000

MODEL_PATH = _FACESYNC_ROOT / "voice" / "models" / "vocal_state_clf.pkl"


class VoiceAnalyzer:
    """
    Continuously records microphone audio and classifies vocal state.

    Usage:
        va = VoiceAnalyzer()
        va.start()
        state, conf = va.vocal_state   # non-blocking read
        va.stop()
    """

    def __init__(self, interval_secs: float = 4.0) -> None:
        """
        Initialise the classifier and prepare the background recording thread.

        Args:
            interval_secs: How often (in seconds) to run a new classification.
        """
        self._interval   = interval_secs
        self._state      = "Unknown"
        self._conf       = 0.0
        self._lock       = threading.Lock()
        self._running    = False
        self._thread: Optional[threading.Thread] = None
        self._clf        = None
        self._sd_available = False

        # Load model
        if not _VOCAL_STATE_AVAILABLE:
            print("[VoiceAnalyzer] vocal_state module not available — voice disabled.")
            return

        try:
            self._clf = VocalStateClassifier(model_path=MODEL_PATH)
            print(f"[VoiceAnalyzer] Model loaded: {MODEL_PATH.name}")
        except ModelNotFoundError:
            print("[VoiceAnalyzer] Model not found — run train_voice_model.py first. Voice disabled.")
            return
        except Exception as exc:
            print(f"[VoiceAnalyzer] Could not load model ({exc}). Voice disabled.")
            return

        # Check sounddevice
        try:
            import sounddevice  # noqa: F401
            self._sd_available = True
        except ImportError:
            print("[VoiceAnalyzer] sounddevice not installed — run: pip install sounddevice. Voice disabled.")

    # ── Thread control ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background audio-capture and classify thread."""
        if self._clf is None or not self._sd_available:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="VoiceAnalyzer")
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Record a fixed-length clip, classify, update state — repeat."""
        import sounddevice as sd
        while self._running:
            try:
                audio = sd.rec(
                    _CHUNK_SAMPLES,
                    samplerate=_SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
                audio = audio.flatten()
                state, conf = self._clf.predict(audio, sr=_SAMPLE_RATE)
                with self._lock:
                    self._state = state.value.capitalize()
                    self._conf  = conf
            except Exception as exc:
                print(f"[VoiceAnalyzer] Error during capture/classify: {exc}")
            # Wait remaining time (recording already took ~interval seconds)
            time.sleep(max(0, self._interval - _DURATION_SECS))

    # ── Public read ───────────────────────────────────────────────────────────

    @property
    def vocal_state(self) -> tuple[str, float]:
        """Return the latest (vocal_state_label, confidence) in a thread-safe way."""
        with self._lock:
            return self._state, self._conf
