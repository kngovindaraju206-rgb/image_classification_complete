# ============================================================
# Image Classification with CNN — TensorFlow/Keras
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import applications, callbacks, layers, models

# ── Config ──────────────────────────────────────────────────
IMAGE_DIR = r"C:\project\dataset\Classification\images"
LABEL_CSV  = r"C:\project\dataset\Classification\image_label.csv"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 25
SEED       = 42
AUTOTUNE   = tf.data.AUTOTUNE


# ── 1. Load & match labels ───────────────────────────────────
df = pd.read_csv(LABEL_CSV)
# Normalize CSV headers to lowercase, then map the known label column to `label`.
df.columns = [c.strip().lower() for c in df.columns]
if "plane" in df.columns and "label" not in df.columns:
    df = df.rename(columns={"plane": "label"})

# Append the image file extension to match the PNG filenames in the image directory.
df["path"] = df["image_name"].apply(lambda n: os.path.join(IMAGE_DIR, f"{n}.png"))
df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

print(f"Dataset size  : {len(df)} images")
print(f"Classes found : {sorted(df['label'].unique())}\n")


# ── 2. Load & preprocess images ──────────────────────────────
def load_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    return np.array(img, dtype=np.float32)

X = np.stack([load_image(p) for p in df["path"]])    # (N, 224, 224, 3)


# ── 3. Encode labels ─────────────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(df["label"])                    # integer codes
num_classes = len(le.classes_)


# ── 4. Train / test split (80:20) ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
)
print(f"Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}\n")


# ── 5. Prepare tf.data pipelines ──────────────────────────────
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.12),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomContrast(0.12),
], name="data_augmentation")


def preprocess_input(image, label):
    image = applications.efficientnet.preprocess_input(image)
    return image, label


def make_dataset(images, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(images), seed=SEED)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(preprocess_input, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(X_train, y_train, training=True)
val_ds = make_dataset(X_val, y_val)
test_ds = make_dataset(X_test, y_test)


# ── 6. CNN model ─────────────────────────────────────────────
def build_model(num_classes):
    base_model = applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling="avg",
    )
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        data_augmentation,
        base_model,
        layers.Dropout(0.3),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model

model, base_model = build_model(num_classes)
model.summary()

# Use class weights to mitigate label imbalance.
class_counts = np.bincount(y_train)
class_weight = {
    i: len(y_train) / (len(class_counts) * count)
    for i, count in enumerate(class_counts)
}

callbacks_list = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks_list,
    verbose=1,
)

# ── 7. Fine-tune the top of the pre-trained network ───────────
base_model.trainable = True
for layer in base_model.layers[:-120]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history.epoch[-1],
    class_weight=class_weight,
    callbacks=callbacks_list,
    verbose=1,
)

# Combine history for plotting
for key in history.history:
    history.history[key] = history.history[key] + history_fine.history.get(key, [])


# ── 7. Training curves ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy"); axes[0].legend()

axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Val")
axes[1].set_title("Loss"); axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=200)
plt.close()


# ── 8. Evaluate on test set ──────────────────────────────────
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Loss     : {test_loss:.4f}\n")


# ── 9. Classification report & confusion matrix ──────────────
y_pred = np.argmax(model.predict(test_ds), axis=1)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.close()


# ── 10. Sample predictions ───────────────────────────────────
num_samples = min(10, len(X_test))
indices     = np.random.choice(len(X_test), num_samples, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

def display_image(image):
    if image.dtype == np.uint8:
        return image
    if image.max() > 1.0:
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image

for ax, idx in zip(axes, indices):
    true_label  = le.classes_[y_test[idx]]
    pred_label  = le.classes_[y_pred[idx]]
    color       = "green" if true_label == pred_label else "red"

    ax.imshow(display_image(X_test[idx]))
    ax.set_title(f"T: {true_label}\nP: {pred_label}", color=color, fontsize=9)
    ax.axis("off")

plt.suptitle("Sample Predictions  (green=correct, red=wrong)", fontsize=12)
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=200)
plt.close()