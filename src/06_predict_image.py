import os
import sys
import json
import numpy as np
import tensorflow as tf
from config import IMG_SIZE, MODELS_DIR, CLASS_NAMES

KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "baseline.keras")
THRESHOLD_FILE = os.path.join(MODELS_DIR, "best_threshold.json")


def load_image(path):
    """
    Load and resize to the same dimensions used during training (IMG_SIZE)
    Args:
        path (str): The image path
    Returns:
        image: The image casted as a tf.float32
    """
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension → shape (1, H, W, 3)
    arr = tf.cast(arr, tf.float32)     # MobileNetV2 preprocess_input expects float32
    return arr


def load_threshold():
    """
    Uses the THRESHOLD_FILE to retrieve the saved threshold. Will default to
    0.5 if not found.
    Returns:
        threshold (float): The threshold for the model
    """
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("threshold", 0.5))
    return 0.5


def load_model_safely():
    """
    Loads to model for prediction using KERAS_MODEL_PATH
    Returns:
        model: The loaded model
    """
    # For inference, compilation is not needed
    return tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/06_predict_image.py <image_path>")
        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    model = load_model_safely()
    threshold = 0.745 if load_threshold() < 0.745 else 0.5
    image = load_image(image_path)
    prob = float(model.predict(image, verbose=0)[0][0])
    pred = 1 if prob >= threshold else 0

    print("Threshold Used:", threshold)
    print("Prediction:", CLASS_NAMES[pred])     # "NORMAL" or "PNEUMONIA"
    print("Probability:", f"{prob:.4f}")

    if pred == 1:
        print("Interpretation: Image is predicted as PNEUMONIA")
    else:
        print("Interpretation: Image is predicted as NORMAL")


if __name__ == "__main__":
    main()