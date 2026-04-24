import os
import sys
import json
import numpy as np
import tensorflow as tf
from config import IMG_SIZE, MODELS_DIR, CLASS_NAMES

TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "int8.tflite") # INT8 model from 05_quantize_tflite.py
THRESHOLD_FILE = os.path.join(MODELS_DIR, "best_threshold.json")


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


def load_image(image_path):
    """
    Load and resize to the same dimensions used during training (IMG_SIZE)
    Args:
        path (str): The image path
    Returns:
        image: The image casted as a tf.float32
    """
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


def quantize_input(image_float, input_details):
    """
    Convert a float32 image to the uint8 format expected by the INT8 model.

    During TFLite INT8 conversion, each tensor is assigned a scale and
    zero_point that map float values to the uint8 range [0, 255].
    Formula: uint8 = float / scale + zero_point

    Args:
        image_float: The casted tf.float32 image
        input_details: The TFLITE_MODEL details
    """
    scale, zero_point = input_details["quantization"]

    if scale == 0:
        raise ValueError("Input scale is 0. Invalid TFLite quantization parameters.")

    image_uint8 = image_float / scale + zero_point
    image_uint8 = np.clip(np.round(image_uint8), 0, 255).astype(input_details["dtype"])
    return image_uint8


def dequantize_output(output_data, output_details):
    """
    Convert the INT8 model's uint8 output back to a float32 probability.

    Inverse of quantize_input: float = scale * (uint8 - zero_point)
    If scale == 0 the output was not quantized (e.g., float fallback), so
    we cast directly.
    Args:
        output_data: The model's quantized raw data
        output_details: The model's details
    Returns:
        out (float32): the models prediction
    """
    scale, zero_point = output_details["quantization"]

    if scale == 0:
        return output_data.astype(np.float32)

    return scale * (output_data.astype(np.float32) - zero_point)


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/06_predict_tflite.py <image_path>")
        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    if not os.path.exists(TFLITE_MODEL_PATH):
        print(f"TFLite model not found: {TFLITE_MODEL_PATH}")
        return

    # Same threshold floor logic as 06_predict_image.py
    threshold = 0.745 if load_threshold() < 0.745 else 0.55

    # TFLite inference
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()  # Must be called before get_*_details()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("TFLite model:", TFLITE_MODEL_PATH)
    print("Threshold Used:", f"{threshold:.3f}")
    print("Input dtype:", input_details["dtype"])   # Expect uint8 for INT8 model
    print("Output dtype:", output_details["dtype"]) # Expect uint8 for INT8 model

    image_float = load_image(image_path)

    # converter.inference_input_type = tf.uint8
    input_data = quantize_input(image_float, input_details)

    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()    # Run the model

    raw_output = interpreter.get_tensor(output_details["index"])
    prob = float(dequantize_output(raw_output, output_details).flatten()[0])

    pred = 1 if prob >= threshold else 0

    print("Probability:", f"{prob:.4f}")
    print("Prediction:", CLASS_NAMES[pred]) # "NORMAL" or "PNEUMONIA"


if __name__ == "__main__":
    main()