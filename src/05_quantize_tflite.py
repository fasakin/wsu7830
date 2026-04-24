import os
import tensorflow as tf
from tensorflow.keras import layers
from config import IMG_SIZE, MODELS_DIR, TFLITE_MODEL_PATH
from config import DATA_DIR
KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "baseline.keras")

def build_inference_model():
    """
    Rebuild the model architecture without the training-time augmentation layer.
    
    The augmentation Sequential block (RandomFlip, RandomRotation, RandomZoom)
    exists only during training and must be stripped before TFLite conversion because
    it is not needed at inference and complicates INT8 quantization.
    Weights are copied from the trained model after construction.

    Returns:
        model: The inference model (mobilenetv2_pneumonia_inference)
    """ 
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights=None,
    )

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="input_image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)     # Scales [0,255] → [-1,1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.40, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier")(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_pneumonia_inference")
    return model

def representative_dataset():
    """
    Produce 100 representative training images for INT8 calibration.

    The TFLite INT8 converter uses these samples to determine the activation
    ranges needed to quantize each layer. Using real training images produces
    better calibration than random noise. batch_size=1 is required by the API.
    Returns:
        image_dataset: The images casted as a float32
    """
 
    train_dir = os.path.join(DATA_DIR, "train")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=1,
        label_mode="binary",
        shuffle=True,
        seed=42,
    ).take(100) # 100 samples is typically sufficient for calibration

    for images, _ in ds:
        yield [tf.cast(images, tf.float32)]

def main():
    if not os.path.exists(KERAS_MODEL_PATH):
        raise FileNotFoundError(f"Could not find trained model: {KERAS_MODEL_PATH}")

    print(f"Loading trained model from: {KERAS_MODEL_PATH}")
    trained_model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)

    print("Building inference-only model without augmentation...")
    inference_model = build_inference_model()
    inference_model.set_weights(trained_model.get_weights())        # Transfer learned weights

    # Full INT8 quantization
    # Setting both input/output types to uint8 means the caller (07_predict_tflite.py)
    # must manually quantize the input image and dequantize the output probability
    # using the scale/zero_point from the interpreter's tensor details.
    print("Converting to TFLite INT8...")
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(TFLITE_MODEL_PATH), exist_ok=True)
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model to: {TFLITE_MODEL_PATH}")

if __name__ == "__main__":
    main()
