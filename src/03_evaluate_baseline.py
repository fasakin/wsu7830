import os
import json
import numpy as np
import tensorflow as tf
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, MODELS_DIR, SEED

KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "baseline.keras")
THRESHOLD_FILE = os.path.join(MODELS_DIR, "best_threshold.json")    # Written by 02_train_baseline.py

def safe_divide(a, b):
    """
    Normal division except it will return 0 if b is 0
    Args:
        a (Number): Numerator
        b (Number): Denominator
    Returns:
        Number (float): The result of the division
    """
    return a / b if b != 0 else 0.0

def load_test_dataset():
    """
    Helper function to load the test dataset located in DATA_DIR
    Returns:
        tuple:  [0] The test dataset\n
                [1] Name of the classes
    """
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,      # Should stay False so label order matches prediction order
    )
    class_names = test_ds.class_names
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    return test_ds, class_names

def collect_labels_and_probs(model, ds):
    """
    Collect ground-truth labels by iterating the dataset before predicting,
    since model.predict() returns probabilities without labels.

    Args: 
        model: The model that will make the predictions
        ds: The dataset in which the model will make the predictions on
    
    Returns:
        tuple:  [0] The true labels
                [1] The models predicted probabilities
    """
    labels = []
    for _, y in ds:
        labels.extend(y.numpy().flatten())
    labels = np.array(labels).astype(int)

    probs = model.predict(ds, verbose=1).flatten()
    return labels, probs

def confusion_counts(labels, preds):
    """
    Calculates the confusion matrix
    Args: 
        labels: The dataset true labels
        preds: The models prediction
    Returns:
        tuple: tn, fp, fn, tp
    """
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return tn, fp, fn, tp

def compute_metrics(labels, preds, probs):
    """
    Calculates all the model performance metrics
    Args:
        labels: The true labels
        preds: The models predictions
        probs: The models prediction probabilities
    Returns:
        dict: Contains all the models performance metrics as key-value pairs
    """
    tn, fp, fn, tp = confusion_counts(labels, preds)

    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)               # Sensitivity: correct PNEUMONIA detections
    specificity = safe_divide(tn, tn + fp)          # Correct NORMAL identifications
    f1 = safe_divide(2 * precision * recall, precision + recall)
    balanced_accuracy = 0.5 * (recall + specificity)  # Key metric for imbalanced data

    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(labels, probs)
    auc = float(auc_metric.result().numpy())

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "auc": auc,
    }

def print_metrics(title, metrics):
    """
    Pretty prints all the metrics in a legible form
    """
    print(f"\n{title}")
    print("Confusion Matrix (rows=Actual, cols=Predicted)")
    print("              Pred 0    Pred 1")
    print(f"Actual 0      {metrics['tn']:6d}    {metrics['fp']:6d}")
    print(f"Actual 1      {metrics['fn']:6d}    {metrics['tp']:6d}")

    print("\nDerived Metrics")
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Precision         : {metrics['precision']:.4f}")
    print(f"Recall            : {metrics['recall']:.4f}")
    print(f"Specificity       : {metrics['specificity']:.4f}")
    print(f"F1-score          : {metrics['f1']:.4f}")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"AUC               : {metrics['auc']:.4f}")

def load_threshold():
    """
    Uses the THRESHOLD_FILE to retrieve the saved threshold. Will default to
    0.5 if not found.
    Returns:
        threshold (float): The threshold for the model
    """
    # Fall back to 0.5 if the threshold file from training is missing
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("threshold", 0.5))
    return 0.5

def main():
    model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)

    threshold = load_threshold()
    print(f"Using threshold: {threshold:.3f}")

    test_ds, class_names = load_test_dataset()
    print("Class names:", class_names)

    print("\nCollecting test predictions...")
    test_labels, test_probs = collect_labels_and_probs(model, test_ds)

    # Apply the validation-tuned threshold instead of the default 0.5
    test_preds = (test_probs >= threshold).astype(int)
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)

    print_metrics("Test Metrics at Saved Threshold", test_metrics)

    # Spot-check a few individual predictions for qualitative review
    print("\nSample Predictions")
    for i in range(min(10, len(test_probs))):
        print(
            f"Sample {i:02d} | True: {test_labels[i]} | "
            f"Pred: {test_preds[i]} | Prob: {test_probs[i]:.4f}"
        )

if __name__ == "__main__":
    main()