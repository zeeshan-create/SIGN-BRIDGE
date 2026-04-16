"""
ANTIGRAVITY ASL — Landmark Model Training Pipeline
===================================================
Trains a dense neural network on MediaPipe hand landmark features.
Produces both .keras and .tflite models for deployment.
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ─── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
INPUT_DIR = os.path.join(PROJECT_DIR, 'processed_data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'landmark_model.keras')
TFLITE_SAVE_PATH = os.path.join(MODELS_DIR, 'landmark_model.tflite')
LABEL_MAP_PATH = os.path.join(MODELS_DIR, 'label_map.json')

# ─── Config ───────────────────────────────────────────────────────────
MAX_SAMPLES_PER_CLASS = 5000
VALIDATION_SPLIT = 0.15
EPOCHS = 3
BATCH_SIZE = 64
IGNORE_DIRS = {'signs_database', '__pycache__', '.git'}


def discover_labels():
    """Dynamically discover sign classes from processed_data/."""
    if not os.path.isdir(INPUT_DIR):
        print(f"[Error] {INPUT_DIR} does not exist. Run preprocessing first.")
        raise SystemExit(1)

    labels = sorted([
        d for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d)) and d not in IGNORE_DIRS
    ])

    if not labels:
        print("[Error] No sign folders found in processed_data/")
        raise SystemExit(1)

    print(f"[Discovery] Found {len(labels)} sign classes: {', '.join(labels)}")
    return labels


def load_data(labels):
    """Load .npy landmark files with balanced augmentation."""
    X, y = [], []
    feature_dim = None

    for label_idx, label in enumerate(labels):
        label_path = os.path.join(INPUT_DIR, label)
        files = [f for f in os.listdir(label_path) if f.endswith('.npy')]

        if not files:
            print(f"  ⚠ {label}: 0 samples (skipped)")
            continue

        # Balanced sampling: Ensure every class has exactly TARGET_SAMPLES
        TARGET_SAMPLES = 1500
        if len(files) < TARGET_SAMPLES:
            files = np.random.choice(files, TARGET_SAMPLES, replace=True)
        else:
            np.random.shuffle(files)
            files = files[:TARGET_SAMPLES]

        loaded = 0
        for file in files:
            try:
                data = np.load(os.path.join(label_path, file))
            except Exception:
                continue

            # Validate feature dimension consistency
            if feature_dim is None:
                feature_dim = data.shape[0]
            elif data.shape[0] != feature_dim:
                continue  # Skip mismatched samples

            X.append(data)
            y.append(label_idx)

            # Augmentation 1: Subtle Gaussian noise
            noisy = data + np.random.normal(0, 0.003, data.shape).astype(np.float32)
            X.append(noisy)
            y.append(label_idx)

            # Augmentation 2: Random scale (simulate distance variation)
            scale = np.random.uniform(0.95, 1.05)
            X.append(data * scale)
            y.append(label_idx)

            loaded += 1

        print(f"  ✓ {label}: {loaded} raw → {loaded * 3} augmented samples")

    if not X:
        print("[Error] No valid samples loaded.")
        raise SystemExit(1)

    print(f"\n[Data] Feature dimension: {feature_dim}")
    print(f"[Data] Total samples: {len(X)}")
    return np.array(X, dtype=np.float32), np.array(y), feature_dim


def build_model(input_dim, num_classes):
    """Build a precision dense network for complex ASL signs."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks():
    """Standard callbacks for perfect training."""
    stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=25, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    return [stop, reduce_lr]


def export_tflite(keras_model):
    """Convert Keras model to TFLite for fast inference."""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(TFLITE_SAVE_PATH, 'wb') as f:
            f.write(tflite_model)

        size_kb = os.path.getsize(TFLITE_SAVE_PATH) / 1024
        print(f"[Export] TFLite model saved: {TFLITE_SAVE_PATH} ({size_kb:.0f} KB)")
    except Exception as e:
        print(f"[Warning] TFLite conversion failed: {e}")


def generate_report(model, X_val, y_val, labels):
    """Print classification report and save confusion matrix."""
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        y_val, y_pred,
        labels=range(len(labels)),
        target_names=labels,
        zero_division=0
    ))

    # Confusion matrix plot
    cm = confusion_matrix(y_val, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, pad=12)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    plot_path = os.path.join(MODELS_DIR, 'confusion_matrix.png')
    fig.savefig(plot_path, dpi=120)
    print(f"[Report] Confusion matrix saved: {plot_path}")
    plt.close(fig)


def train():
    """Full training pipeline."""
    labels = discover_labels()

    # Save label map
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump({'labels': labels}, f, indent=2)
    print(f"[Sync] Label map saved: {LABEL_MAP_PATH}")

    # Load data
    X, y, feature_dim = load_data(labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )

    # Build & train
    model = build_model(feature_dim, len(labels))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=12, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    print(f"\n[Training] {len(labels)} classes, {len(X_train)} train / {len(X_val)} val samples")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Save
    model.save(MODEL_SAVE_PATH)
    print(f"\n[Saved] Keras model: {MODEL_SAVE_PATH}")

    # Export TFLite
    export_tflite(model)

    # Generate performance report
    generate_report(model, X_val, y_val, labels)

    print("\n✓ Training complete.")


if __name__ == "__main__":
    train()
