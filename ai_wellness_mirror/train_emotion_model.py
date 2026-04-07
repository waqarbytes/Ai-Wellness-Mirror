"""
train_emotion_model.py
======================
Train a lightweight CNN on the FER2013 dataset and save the model to
models/emotion_model.keras.

Usage:
    python3 train_emotion_model.py                    # full 30-epoch run
    python3 train_emotion_model.py --epochs 5         # quick test run
    python3 train_emotion_model.py --csv /path/to/fer2013.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Suppress TF info / warning logs ────────────────────────────────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# ── Constants ───────────────────────────────────────────────────────────────
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMG_SIZE       = 48
NUM_CLASSES    = 7
BATCH_SIZE     = 64

PROJECT_ROOT   = Path(__file__).resolve().parent
DEFAULT_CSV    = PROJECT_ROOT.parent / "fer2013.csv"   # Desktop/Facesync/fer2013.csv
MODEL_OUT      = PROJECT_ROOT / "models" / "emotion_model.keras"


# ── Data loading ─────────────────────────────────────────────────────────────
def load_split(df: pd.DataFrame, usage: str):
    """Parse pixels column → (N, 48, 48, 1) float32 array."""
    sub = df[df["Usage"] == usage].copy()
    X = np.array(
        [np.array(p.split(), dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE, 1)
         for p in sub["pixels"]],
        dtype=np.float32,
    )
    X /= 255.0
    y = sub["emotion"].values.astype(np.int32)
    return X, y


# ── Model ────────────────────────────────────────────────────────────────────
def build_model() -> keras.Model:
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        layers.RandomTranslation(0.08, 0.08),
    ], name="augmentation")

    def conv_block(x, filters, dropout=0.25):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(dropout)(x)
        return x

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = aug(inputs)
    x = conv_block(x, 32,  dropout=0.20)   # → 24×24
    x = conv_block(x, 64,  dropout=0.25)   # → 12×12
    x = conv_block(x, 128, dropout=0.30)   # →  6×6
    x = conv_block(x, 256, dropout=0.35)   # →  3×3
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="fer_emotion_cnn")
    return model


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train FER2013 Emotion CNN")
    parser.add_argument("--csv",    type=str, default=str(DEFAULT_CSV),
                        help="Path to fer2013.csv")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default 30)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found at: {csv_path}")
        print("  Pass the correct path with:  --csv /path/to/fer2013.csv")
        sys.exit(1)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*56}")
    print("  FER2013 Emotion Model Training")
    print(f"{'='*56}")
    print(f"  CSV     : {csv_path}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Model → : {MODEL_OUT}")
    print(f"  TF ver  : {tf.__version__}")
    print(f"{'='*56}\n")

    # ── Load data ────────────────────────────────────────────────────────────
    print("[1/5] Loading FER2013 CSV …")
    df = pd.read_csv(csv_path)
    print(f"      Total samples : {len(df):,}")
    print(f"      Columns       : {list(df.columns)}")

    print("\n[2/5] Parsing pixel arrays …")
    X_train, y_train = load_split(df, "Training")
    X_val,   y_val   = load_split(df, "PublicTest")
    X_test,  y_test  = load_split(df, "PrivateTest")

    print(f"      Train : {X_train.shape}  ({len(X_train):,} samples)")
    print(f"      Val   : {X_val.shape}   ({len(X_val):,} samples)")
    print(f"      Test  : {X_test.shape}   ({len(X_test):,} samples)")

    # Emotion distribution
    print("\n      Emotion distribution (train):")
    for i, label in enumerate(EMOTION_LABELS):
        count = int((y_train == i).sum())
        print(f"        {i} {label:<10}: {count:>5,}")

    # ── Class weights (handles Disgust imbalance) ─────────────────────────
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_train,
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\n      Class weights (for imbalance): {[f'{w:.2f}' for w in class_weights]}")

    # ── One-hot encode ────────────────────────────────────────────────────
    y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_oh   = keras.utils.to_categorical(y_val,   NUM_CLASSES)

    # ── Build & compile model ─────────────────────────────────────────────
    print("\n[3/5] Building model …")
    model = build_model()
    model.summary(line_length=70)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4,
            verbose=1, min_lr=1e-6,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_OUT),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n[4/5] Training for up to {args.epochs} epochs …")
    print("      (Early stopping will kick in if val_accuracy stops improving)\n")

    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate on test set ──────────────────────────────────────────────
    print("\n[5/5] Evaluating on PrivateTest set …")
    best_model = keras.models.load_model(str(MODEL_OUT))
    y_pred_probs = best_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_loss, test_acc = best_model.evaluate(X_test,
                                              keras.utils.to_categorical(y_test, NUM_CLASSES),
                                              batch_size=BATCH_SIZE, verbose=0)

    print(f"\n{'='*56}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"{'='*56}\n")
    print(classification_report(y_test, y_pred, target_names=EMOTION_LABELS))
    print(f"\n✅ Model saved to: {MODEL_OUT}")
    print("   Run main.py — it will automatically use this model.\n")


if __name__ == "__main__":
    main()
