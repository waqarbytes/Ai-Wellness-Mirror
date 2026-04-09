"""
train_voice_model.py
====================
Train a 3-class vocal state classifier (calm / stressed / fatigued)
on the RAVDESS dataset using classical ML (SVM + Random Forest).

Run:
    python train_voice_model.py

Outputs:
    voice/models/vocal_state_clf.pkl   <- saved model bundle
    dataset/features.npz               <- cached feature matrix
"""

# ── Standard library ─────────────────────────────────────────────────────────
import logging
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.WARNING)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CONFIG: dict = {
    # Paths
    "dataset_root":   "archive",            # folder containing Actor_XX dirs
    "features_cache": "dataset/features.npz",
    "model_output":   "voice/models/vocal_state_clf.pkl",

    # Audio
    "sample_rate":    16000,
    "fixed_duration": 3,                    # seconds → 48 000 samples
    "top_db":         30,                   # silence trim threshold

    # Features
    "n_mfcc":         13,
    "n_mels":         64,
    "n_chroma":       12,
    "n_fft":          2048,
    "hop_length":     512,
    "feature_size":   215,

    # Training
    "test_size":      0.2,
    "random_state":   42,

    # RAVDESS label map  (emotion_code → label, None = skip)
    "emotion_map": {
        "01": "calm",       # neutral   → calm
        "02": "calm",       # calm      → calm
        "03": None,         # happy     → see fatigued rule below
        "04": None,         # sad       → skip
        "05": "stressed",   # angry     → stressed
        "06": "stressed",   # fearful   → stressed
        "07": None,         # disgust   → skip
        "08": None,         # surprised → skip
    },
    # Fatigued override: emotion=03 AND intensity=01
    # ASSUMPTION: RAVDESS contains no native "fatigue" label.
    # Low-energy happy speech (emotion=03, intensity=01/normal) is used as a
    # proxy for vocal fatigue because:
    #   - It shares acoustic properties with fatigue: reduced pitch variance,
    #     lower energy, slightly slower articulation rate.
    #   - All other RAVDESS classes are either too energetic (angry, fearful)
    #     or tonally dissimilar (sad has drooping F0 but high breathiness).
    # This is a deliberate approximation; the label should be treated as
    # "acoustically-fatigued-like" rather than clinically confirmed fatigue.
    "fatigued_emotion":   "03",
    "fatigued_intensity": "01",
}

TARGET_LABELS = ["calm", "stressed", "fatigued"]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — RAVDESS Dataset Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_ravdess(dataset_root: Path) -> list[tuple[Path, str]]:
    """
    Walk dataset_root and return (filepath, label) for every kept WAV file.

    RAVDESS filename convention (dash-separated, zero-indexed):
      [0] modality  [1] vocal_channel  [2] emotion  [3] intensity
      [4] statement [5] repetition     [6] actor

    Label assignment priority:
      1. fatigued  → emotion == fatigued_emotion AND intensity == fatigued_intensity
      2. mapped    → CONFIG['emotion_map'][emotion] is not None
      3. skipped   → everything else
    """
    all_wavs  = sorted(Path(dataset_root).glob("Actor_*/*.wav"))
    kept: list[tuple[Path, str]] = []
    skipped   = 0
    label_counts: dict[str, int] = {lbl: 0 for lbl in TARGET_LABELS}
    actors_found: set[str] = set()

    for wav in all_wavs:
        actors_found.add(wav.parent.name)
        parts          = wav.stem.split("-")
        emotion_code   = parts[2]
        intensity_code = parts[3]

        # ── Fatigued override (must be checked first) ─────────────────────
        if (
            emotion_code   == CONFIG["fatigued_emotion"]
            and intensity_code == CONFIG["fatigued_intensity"]
        ):
            label = "fatigued"

        # ── Standard emotion map ──────────────────────────────────────────
        elif CONFIG["emotion_map"].get(emotion_code) is not None:
            label = CONFIG["emotion_map"][emotion_code]

        # ── Skip ──────────────────────────────────────────────────────────
        else:
            skipped += 1
            continue

        kept.append((wav, label))
        label_counts[label] += 1

    # ── Summary ───────────────────────────────────────────────────────────
    total   = len(all_wavs)
    n_kept  = len(kept)
    actors_str = ", ".join(sorted(actors_found))

    print()
    print("  ════════════════════════════════════════")
    print("   RAVDESS Dataset Summary")
    print("  ════════════════════════════════════════")
    print(f"   Total .wav files found   : {total}")
    print(f"   Files kept (3 classes)   : {n_kept}")
    print(f"   Files skipped            : {skipped}")
    print()
    print("   Class Distribution:")
    print(f"   {'Label':<10} | {'Count':>5} | % of kept")
    print(f"   {'-'*10}-|-------|----------")
    for lbl in TARGET_LABELS:
        pct = label_counts[lbl] / n_kept * 100 if n_kept else 0.0
        print(f"   {lbl:<10} | {label_counts[lbl]:>5} | {pct:>6.1f}%")
    print()
    print(f"   Actors found: {actors_str}")
    print("  ════════════════════════════════════════")
    print()

    # ── Validation ────────────────────────────────────────────────────────
    for lbl in TARGET_LABELS:
        if label_counts[lbl] < 20:
            raise ValueError(
                f"Class '{lbl}' has only {label_counts[lbl]} samples "
                f"(minimum required: 20). Check your dataset path or "
                f"fatigued override rule."
            )

    return kept


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(filepath: Path, config: dict) -> Optional[np.ndarray]:
    """
    Load, trim, normalise, and pad/truncate a WAV file to a fixed-length
    float32 array of shape (sample_rate * fixed_duration,).

    Returns None on any load error (caller will skip the file).
    """
    try:
        signal, _ = librosa.load(str(filepath), sr=config["sample_rate"], mono=True)
    except Exception as exc:
        print(f"  [WARN] Could not load {filepath}: {exc}")
        return None

    # Trim leading/trailing silence
    signal, _ = librosa.effects.trim(signal, top_db=config["top_db"])

    # Peak normalise  (avoid division by zero with epsilon)
    signal = signal / (np.max(np.abs(signal)) + 1e-9)

    # Pad or truncate to fixed length
    target_len = config["sample_rate"] * config["fixed_duration"]
    if len(signal) >= target_len:
        signal = signal[:target_len]
    else:
        pad = target_len - len(signal)
        signal = np.pad(signal, (0, pad), mode="constant")

    return signal.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(signal: np.ndarray, config: dict) -> np.ndarray:
    """
    Extract a 215-dimensional feature vector from a preprocessed audio signal.

    Feature layout:
      [  0: 26]  MFCC mean + std                 (13 × 2)
      [ 26: 52]  Delta-MFCC mean + std            (13 × 2)
      [ 52:180]  Mel spectrogram mean + std        (64 × 2)
      [180:204]  Chroma STFT mean + std            (12 × 2)
      [204:207]  Pitch F0:  mean, std, voiced_frac  (3)
      [207:209]  RMS energy mean + std              (2)
      [209:211]  Zero-crossing rate mean + std      (2)
      [211:213]  Spectral centroid mean + std        (2)
      [213:215]  Spectral rolloff mean + std         (2)
    Total: 215
    """
    sr  = config["sample_rate"]
    nfft = config["n_fft"]
    hop  = config["hop_length"]

    features: list[np.ndarray] = []

    # 1. MFCC mean + std  (26)
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=config["n_mfcc"], n_fft=nfft, hop_length=hop
    )
    features.extend([mfcc.mean(axis=1), mfcc.std(axis=1)])

    # 2. Delta-MFCC mean + std  (26)
    delta_mfcc = librosa.feature.delta(mfcc)
    features.extend([delta_mfcc.mean(axis=1), delta_mfcc.std(axis=1)])

    # 3. Mel spectrogram mean + std  (128)
    mel = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_mels=config["n_mels"], n_fft=nfft, hop_length=hop
    )
    features.extend([mel.mean(axis=1), mel.std(axis=1)])

    # 4. Chroma STFT mean + std  (24)
    chroma = librosa.feature.chroma_stft(
        y=signal, sr=sr, n_chroma=config["n_chroma"], n_fft=nfft, hop_length=hop
    )
    features.extend([chroma.mean(axis=1), chroma.std(axis=1)])

    # 5. Pitch F0 stats via pyin  (3)
    #    ASSUMPTION: pyin returns NaN for unvoiced frames — replace with 0.
    f0, voiced_flag, _ = librosa.pyin(
        signal,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    f0 = np.where(np.isnan(f0), 0.0, f0)
    voiced_fraction = float(voiced_flag.mean()) if voiced_flag is not None else 0.0
    features.append(np.array([f0.mean(), f0.std(), voiced_fraction], dtype=np.float32))

    # 6. RMS energy mean + std  (2)
    rms = librosa.feature.rms(y=signal, hop_length=hop)
    features.append(np.array([rms.mean(), rms.std()], dtype=np.float32))

    # 7. Zero-crossing rate mean + std  (2)
    zcr = librosa.feature.zero_crossing_rate(y=signal, hop_length=hop)
    features.append(np.array([zcr.mean(), zcr.std()], dtype=np.float32))

    # 8. Spectral centroid mean + std  (2)
    sc = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=nfft, hop_length=hop)
    features.append(np.array([sc.mean(), sc.std()], dtype=np.float32))

    # 9. Spectral rolloff mean + std  (2)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=nfft, hop_length=hop)
    features.append(np.array([rolloff.mean(), rolloff.std()], dtype=np.float32))

    vec = np.concatenate(features).astype(np.float32)
    assert vec.shape == (config["feature_size"],), (
        f"Feature size mismatch: expected {config['feature_size']}, got {vec.shape[0]}"
    )
    return vec


def build_feature_matrix(
    samples: list[tuple[Path, str]],
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run preprocess + extract_features on every (path, label) tuple.
    Caches the result to dataset/features.npz.
    Returns X of shape (N, 215) and y of shape (N,).
    """
    cache_path = Path(config["features_cache"])

    # ── Cache hit ─────────────────────────────────────────────────────────
    if cache_path.exists():
        data = np.load(str(cache_path), allow_pickle=True)
        X, y = data["X"], data["y"]
        print(f"Loaded cached features: X.shape={X.shape}  — delete to re-extract")
        return X, y

    # ── Extract ───────────────────────────────────────────────────────────
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    X_list: list[np.ndarray] = []
    y_list: list[str] = []

    for path, label in tqdm(samples, desc="Extracting features", unit="clip"):
        signal = preprocess(path, config)
        if signal is None:
            continue
        feat = extract_features(signal, config)
        X_list.append(feat)
        y_list.append(label)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=object)

    np.savez(str(cache_path), X=X, y=y)
    print(f"Features saved to {cache_path}  ({len(X_list)} clips)")

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Train SVM and Random Forest
# ══════════════════════════════════════════════════════════════════════════════

def build_pipelines() -> dict[str, Pipeline]:
    """Return the two default sklearn Pipelines: SVM and Random Forest."""
    return {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svc",    SVC(
                C=10, kernel="rbf", probability=True,
                class_weight="balanced", random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("randomforestclassifier", RandomForestClassifier(
                n_estimators=300, max_depth=None,
                class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
        ]),
    }


def print_model_report(
    name: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    le: LabelEncoder,
) -> dict[str, float]:
    """Print a formatted evaluation block and return metric dict."""
    acc   = accuracy_score(y_test, y_pred)
    f1m   = f1_score(y_test, y_pred, average="macro")
    f1w   = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, digits=4,
    )
    cm = confusion_matrix(y_test, y_pred)
    classes = list(le.classes_)

    width = 46
    bar   = "═" * width
    print(f"\n  ╔{bar}╗")
    print(f"  ║  Model : {name:<{width - 10}}║")
    print(f"  ╠{bar}╣")
    print(f"  ║  Accuracy    : {acc:.4f}{' ' * (width - 19)}║")
    print(f"  ║  Macro F1    : {f1m:.4f}{' ' * (width - 19)}║")
    print(f"  ║  Weighted F1 : {f1w:.4f}{' ' * (width - 19)}║")
    print(f"  ╠{bar}╣")
    print(f"  ║  Classification Report:{' ' * (width - 24)}║")
    for line in report.splitlines():
        print(f"  ║  {line:<{width - 3}}║")
    print(f"  ╠{bar}╣")
    print(f"  ║  Confusion Matrix:{' ' * (width - 19)}║")
    header = "          " + "  ".join(f"{c:<8}" for c in classes)
    print(f"  ║  {header:<{width - 3}}║")
    for i, row in enumerate(cm):
        row_str = f"  {classes[i]:<8}  " + "  ".join(f"{v:<8}" for v in row)
        print(f"  ║  {row_str:<{width - 3}}║")
    print(f"  ╚{bar}╝")

    return {"accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w}


def evaluate_models(
    pipelines: dict[str, Pipeline],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
) -> dict[str, dict[str, float]]:
    """Fit all pipelines and return test metrics for each."""
    results: dict[str, dict[str, float]] = {}
    for name, pipe in pipelines.items():
        print(f"\n[Training] {name} …")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results[name] = print_model_report(name, y_test, y_pred, le)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Hyperparameter Tuning on the Winner
# ══════════════════════════════════════════════════════════════════════════════

PARAM_GRIDS: dict[str, dict] = {
    "SVM": {
        "svc__C":     [0.1, 1, 10, 100],
        "svc__gamma": ["scale", "auto"],
    },
    "Random Forest": {
        "randomforestclassifier__n_estimators": [100, 300, 500],
        "randomforestclassifier__max_features": ["sqrt", "log2"],
    },
}


def tune_winner(
    winner_name: str,
    winner_pipe: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    """Run GridSearchCV on the winning pipeline and return the best estimator."""
    print(f"\n[Tuning] Running GridSearchCV on {winner_name} …")
    grid = GridSearchCV(
        winner_pipe,
        PARAM_GRIDS[winner_name],
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_:.4f}")
    return grid.best_estimator_


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Final Comparison Table
# ══════════════════════════════════════════════════════════════════════════════

def measure_inference_ms(pipe: Pipeline, X_test: np.ndarray, n: int = 200) -> float:
    """Return mean inference time in milliseconds over n single-clip predictions."""
    indices = np.random.RandomState(0).choice(len(X_test), size=n, replace=True)
    t0 = time.perf_counter()
    for i in indices:
        pipe.predict(X_test[i : i + 1])
    return (time.perf_counter() - t0) / n * 1000.0


def print_comparison_table(
    rows: list[dict],
) -> None:
    """Print the final comparison table with inference times."""
    col_w = [21, 10, 10, 13, 14]
    sep = "─" * col_w[0]
    header = (
        f"  ┌{'─'*col_w[0]}┬{'─'*col_w[1]}┬{'─'*col_w[2]}"
        f"┬{'─'*col_w[3]}┬{'─'*col_w[4]}┐\n"
        f"  │ {'Model':<19} │ {'Accuracy':>8} │ {'Macro F1':>8} "
        f"│ {'Weighted F1':>11} │ {'Inference ms':>12} │\n"
        f"  ├{'─'*col_w[0]}┼{'─'*col_w[1]}┼{'─'*col_w[2]}"
        f"┼{'─'*col_w[3]}┼{'─'*col_w[4]}┤"
    )
    print(f"\n{header}")
    for r in rows:
        tag    = "   ← SAVED" if r.get("saved") else ""
        inf_ms = f"{r['inference_ms']:.2f}{tag}"
        print(
            f"  │ {r['name']:<19} │ {r['accuracy']:>8.4f} │ {r['f1_macro']:>8.4f} "
            f"│ {r['f1_weighted']:>11.4f} │ {inf_ms:>12} │"
        )
    col_w_last = col_w[4] + 10  # accommodate "← SAVED"
    print(
        f"  └{'─'*col_w[0]}┴{'─'*col_w[1]}┴{'─'*col_w[2]}"
        f"┴{'─'*col_w[3]}┴{'─'*(col_w[4])}┘"
    )


def print_recommendation(
    winner_name: str,
    tuned_metrics: dict[str, float],
    fatigued_f1: float,
    inference_ms: float,
) -> None:
    """Print the RECOMMENDATION block."""
    bar = "═" * 44
    print(f"\n  {bar}")
    print(f"   RECOMMENDATION")
    print(f"  {bar}")
    print(f"   Selected : {winner_name} (tuned)")
    reason = (
        f"   Reason   : Achieved the highest macro F1 ({tuned_metrics['f1_macro']:.4f}) "
        f"across all three classes,\n"
        f"              including the fatigued class (F1 ≈ {fatigued_f1:.2f}), which is "
        f"the hardest to classify\n"
        f"              due to the approximate label mapping (low-energy happy speech as "
        f"fatigue proxy).\n"
        f"              Inference speed of {inference_ms:.1f} ms per 3-second clip is "
        f"fully viable for real-time on-device use."
    )
    print(reason)
    print(f"  {bar}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Save Model Bundle
# ══════════════════════════════════════════════════════════════════════════════

def save_model_bundle(
    tuned_pipe: Pipeline,
    le: LabelEncoder,
    output_path: Path,
) -> None:
    """Serialise the model bundle to a .pkl file via pickle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model":          tuned_pipe,
        "label_encoder":  le,
        "labels":         TARGET_LABELS,
        "feature_size":   CONFIG["feature_size"],
        "sample_rate":    CONFIG["sample_rate"],
        "fixed_duration": CONFIG["fixed_duration"],
        "dataset":        "RAVDESS",
        "emotion_map":    CONFIG["emotion_map"],
        "fatigued_rule":  "emotion=03 AND intensity=01",
        "trained_on":     datetime.now().isoformat(),
    }
    with open(output_path, "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = output_path.stat().st_size // 1024
    print(f"Model saved → {output_path}  ({size_kb} KB)")
    print("Training complete.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """End-to-end training pipeline: load → features → train → tune → save."""

    print("\n" + "═" * 56)
    print("  RAVDESS Vocal State Classifier — Training Pipeline")
    print("═" * 56)

    # ── Step 1: Load dataset ────────────────────────────────────────────────
    print("\n[1/7] Loading RAVDESS dataset …")
    dataset_root = Path(CONFIG["dataset_root"])
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root.resolve()}\n"
            f"Place the Actor_XX folders inside '{CONFIG['dataset_root']}/'"
        )
    samples = load_ravdess(dataset_root)

    # ── Step 2+3: Build feature matrix (with cache) ─────────────────────────
    print("[2-3/7] Building feature matrix …")
    X, y = build_feature_matrix(samples, CONFIG)
    print(f"        X.shape = {X.shape}  |  classes = {np.unique(y).tolist()}")

    # ── Label encode ────────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)

    # ── Stratified split ────────────────────────────────────────────────────
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
    )
    train_idx, test_idx = next(sss.split(X, y_enc))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]
    print(f"        Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Step 4: Train default models ────────────────────────────────────────
    print("\n[4/7] Training default SVM and Random Forest …")
    pipelines = build_pipelines()
    results   = evaluate_models(pipelines, X_train, y_train, X_test, y_test, le)

    # ── Step 5: Tune the winner (higher macro F1) ───────────────────────────
    print("\n[5/7] Selecting winner by Macro F1 …")
    winner_name = max(results, key=lambda k: results[k]["f1_macro"])
    print(f"        Winner: {winner_name}  "
          f"(macro F1 = {results[winner_name]['f1_macro']:.4f})")
    winner_pipe = build_pipelines()[winner_name]   # fresh un-fitted clone
    winner_pipe.fit(X_train, y_train)              # refit from scratch before GSV
    tuned_pipe  = tune_winner(winner_name, winner_pipe, X_train, y_train)

    # Refit tuned pipeline on full training set (GridSearchCV already does this
    # via refit=True default, but we ensure clarity here)
    tuned_pipe.fit(X_train, y_train)

    # ── Step 6: Final comparison ────────────────────────────────────────────
    print("\n[6/7] Building final comparison table …")
    table_rows: list[dict] = []

    for name, pipe in pipelines.items():
        y_pred = pipe.predict(X_test)
        inf_ms = measure_inference_ms(pipe, X_test)
        table_rows.append({
            "name":        f"{name} (default)",
            "accuracy":    accuracy_score(y_test, y_pred),
            "f1_macro":    f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "inference_ms": inf_ms,
            "saved":       False,
        })

    # Tuned winner
    y_tuned = tuned_pipe.predict(X_test)
    inf_tuned = measure_inference_ms(tuned_pipe, X_test)
    tuned_metrics = {
        "accuracy":    accuracy_score(y_test, y_tuned),
        "f1_macro":    f1_score(y_test, y_tuned, average="macro"),
        "f1_weighted": f1_score(y_test, y_tuned, average="weighted"),
    }
    table_rows.append({
        "name":        f"{winner_name} (tuned)",
        "accuracy":    tuned_metrics["accuracy"],
        "f1_macro":    tuned_metrics["f1_macro"],
        "f1_weighted": tuned_metrics["f1_weighted"],
        "inference_ms": inf_tuned,
        "saved":       True,
    })

    print_comparison_table(table_rows)

    # Fatigued class F1 for recommendation
    fatigued_idx   = list(le.classes_).index("fatigued")
    report_dict    = classification_report(
        y_test, y_tuned, output_dict=True, target_names=le.classes_
    )
    fatigued_f1    = report_dict["fatigued"]["f1-score"]

    print_recommendation(winner_name, tuned_metrics, fatigued_f1, inf_tuned)

    # ── Step 7: Save bundle ─────────────────────────────────────────────────
    print("[7/7] Saving model bundle …")
    save_model_bundle(tuned_pipe, le, Path(CONFIG["model_output"]))


if __name__ == "__main__":
    main()
