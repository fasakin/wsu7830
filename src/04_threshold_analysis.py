import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, MODELS_DIR

MODEL_PATH = os.path.join(MODELS_DIR, "baseline.keras")
PLOT_PATH = os.path.join(MODELS_DIR, "test_threshold_analysis.png")


def safe_div(a, b):
    """
    Normal division except it will return 0 if b is 0
    Args:
        a (Number): Numerator
        b (Number): Denominator
    Returns:
        Number (float): The result of the division
    """
    return a / b if b != 0 else 0.0


def load_model_for_inference():
    """
    Loads the model from MODEL_PATH
    Returns:
        model: The saved model
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # compile=False avoids errors from custom training loss/metrics
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


def load_test_dataset():
    """
    Helper function to load the test dataset located in DATA_DIR
    Returns:
        tuple:  [0] The test dataset\n
                [1] Name of the classes
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )
    class_names = ds.class_names
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, class_names


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
    labels = np.array(labels, dtype=int)

    probs = model.predict(ds, verbose=1).flatten()
    return labels, probs


def evaluate_threshold(labels, probs, threshold):
    """
    Tests the threshold to see what the model performance metrics are
    Args: 
        labels: The dataset true labels
        probs: The models predicted probabilities
        threshold: The point in which a probability is 1 or 0
    Returns:
        dict: A dictionary containing all performance metrics 
    """
    # Compute all classification metrics for a single threshold value
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    bal_acc = 0.5 * (recall + specificity)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": bal_acc,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    model = load_model_for_inference()
    test_ds, class_names = load_test_dataset()

    print("Class names:", class_names)

    labels, probs = collect_labels_and_probs(model, test_ds)
    
    # Threshold sweep on the TEST set (informational only)
    thresholds = np.linspace(0.10, 0.90, 200)

    results = [evaluate_threshold(labels, probs, t) for t in thresholds]
    best_result = max(results, key=lambda x: x["balanced_accuracy"])

    print("\nBEST TEST THRESHOLD (analysis only)")
    print(f"Threshold          : {best_result['threshold']:.3f}")
    print(f"Balanced Accuracy  : {best_result['balanced_accuracy']:.4f}")
    print(f"Accuracy           : {best_result['accuracy']:.4f}")
    print(f"Precision          : {best_result['precision']:.4f}")
    print(f"Recall             : {best_result['recall']:.4f}")
    print(f"Specificity        : {best_result['specificity']:.4f}")
    print(
        f"Confusion Matrix   : [[{best_result['tn']}, {best_result['fp']}], "
        f"[{best_result['fn']}, {best_result['tp']}]]"
    )

    bal_accs = [r["balanced_accuracy"] for r in results]
    recalls = [r["recall"] for r in results]
    specs = [r["specificity"] for r in results]
    precisions = [r["precision"] for r in results]

    # Plot: metric curves vs threshold
    # The vertical dashed line marks the threshold that maximises balanced
    # accuracy on the test set. The trade-off between recall and specificity
    # is visible where the two curves cross.
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, bal_accs, label="Balanced Accuracy")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, specs, label="Specificity")
    plt.plot(thresholds, precisions, label="Precision")
    plt.axvline(
        best_result["threshold"],
        linestyle="--",
        label=f"Best t={best_result['threshold']:.2f}",
    )

    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Test Threshold Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print(f"Saved plot to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
