import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from config import (
    DATA_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    VAL_SPLIT,
    SEED,
    BASELINE_MODEL_DIR,
    MODELS_DIR,
)

INITIAL_LEARNING_RATE = 1e-3   # Used while the MobileNetV2 base is frozen
FINE_TUNE_LEARNING_RATE = 1e-5 # Much smaller LR to avoid destroying pretrained weights
FINE_TUNE_EPOCHS = 3           # Additional epochs with top layers unfrozen

# reduce over-predicting pneumonia
ALPHA = 0.50
GAMMA = 2.0                    # Focuses loss on hard / misclassified examples

THRESHOLD_FILE = os.path.join(MODELS_DIR, "best_threshold.json")
THRESHOLD_SWEEP_FILE = os.path.join(MODELS_DIR, "threshold_sweep.json")
ROC_FILE = os.path.join(MODELS_DIR, "roc_data.npz")


def list_image_files(folder:str):
    """
    Collects the paths for all the images in the folder specified
    Args:
        folder (str): The folder to look for the images
    Returns:
        list[str]: A sorted list containing the paths for all the images
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(valid_exts):
            files.append(path)
    return sorted(files)


def stratified_split():
    """
    Split the training folder into train/val while preserving class ratios.

    Returns:
        tuple:  [0] Paths for training + testing (list[str])\n
                [1] Labels used for training (float32)\n
                [2] Paths for the Validation set (list[str])\n
                [3] Validation set labels (float32)
    """
    # Since the chest X-ray dataset is imbalanced (~3.9× more PNEUMONIA than NORMAL),
    # a random split will be unrepresentative of the validation set. Using stratified split
    # ensure each class is shuffled independently before slicing VAL_SPLIT off the top.
    
    normal_folder = os.path.join(DATA_DIR, "train", "NORMAL")
    pneumonia_folder = os.path.join(DATA_DIR, "train", "PNEUMONIA")

    normal_files = list_image_files(normal_folder)
    pneumonia_files = list_image_files(pneumonia_folder)

    rng = np.random.default_rng(SEED) # For consistent results
    rng.shuffle(normal_files)
    rng.shuffle(pneumonia_files)

    n_normal_val = max(1, int(len(normal_files) * VAL_SPLIT))
    n_pneumonia_val = max(1, int(len(pneumonia_files) * VAL_SPLIT))

    val_normal = normal_files[:n_normal_val]
    train_normal = normal_files[n_normal_val:]

    val_pneumonia = pneumonia_files[:n_pneumonia_val]
    train_pneumonia = pneumonia_files[n_pneumonia_val:]

    # Labels: 0 = NORMAL, 1 = PNEUMONIA (matches CLASS_NAMES index in config)
    train_paths = train_normal + train_pneumonia
    train_labels = [0] * len(train_normal) + [1] * len(train_pneumonia)

    val_paths = val_normal + val_pneumonia
    val_labels = [0] * len(val_normal) + [1] * len(val_pneumonia)

    train_combined = list(zip(train_paths, train_labels))
    val_combined = list(zip(val_paths, val_labels))

    rng.shuffle(train_combined)
    rng.shuffle(val_combined)

    train_paths, train_labels = zip(*train_combined)
    val_paths, val_labels = zip(*val_combined)

    return (
        list(train_paths),
        np.array(train_labels, dtype=np.float32),
        list(val_paths),
        np.array(val_labels, dtype=np.float32),
    )


def make_dataset_from_paths(paths:list[str], labels:list, shuffle=False):
    """
    Build a tf.data pipeline from file paths and float32 labels.
    Args:
        paths (list[str]): Paths to the data
        labels (list): Labels for the data
        shuffle (bool): Weather the data should be shuffled
    Returns:
        ds (tf.data.Dataset): The newly created dataset
    """

    # The images are decoded, resized to IMG_SIZE, and cast to float32 (raw pixel
    # values 0-255). MobileNetV2 preprocessing (scaling to [-1, 1]) is applied
    # inside the model graph so that the same model works correctly at inference.

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    def load_image(path:str, label):
        """
        Loading the image into tensorflow
        Args:
            path (str): Image path
            label: The label for the image
        Returns:
            tuple:  [0] The loaded image\n
                    [1] The label of the image (float32)
        """
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32)
        return image, tf.cast(label, tf.float32)

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_test_dataset():
    """
    Test set is loaded separately (never seen during training or threshold selection)
    Returns:
        tuple:  [0] The test dataset\n
                [1] Test dataset class labels
    """
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,                # Keep order deterministic for metric alignment
    )
    class_names = test_ds.class_names
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
    return test_ds, class_names


def inspect_labels(labels, name="dataset"):
    """
    Prints the number of NORMAL and PNEUMONIA label distribution
    Args:
        labels (list[int]): List holding the labels
        name (str): Name of the dataset
    """
    labels = np.array(labels).astype(int)
    n_normal = int(np.sum(labels == 0))
    n_pneumonia = int(np.sum(labels == 1))

    print(f"\n{name} label distribution")
    print(f"NORMAL     : {n_normal}")
    print(f"PNEUMONIA  : {n_pneumonia}")

class BalancedAccuracy(tf.keras.metrics.Metric):
    # Custom Keras metric: (Recall + Specificity) / 2.

    # Primary monitor for EarlyStopping and ModelCheckpoint because
    # plain accuracy is misleading on the imbalanced chest X-ray dataset.


    def __init__(self, name="balanced_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.tn = self.add_weight(name="tn", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]) >= 0.5, tf.float32)

        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1.0), tf.equal(y_pred, 1.0)), tf.float32))
        tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0.0), tf.equal(y_pred, 0.0)), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0.0), tf.equal(y_pred, 1.0)), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1.0), tf.equal(y_pred, 0.0)), tf.float32))

        self.tp.assign_add(tp)
        self.tn.assign_add(tn)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        recall = self.tp / (self.tp + self.fn + 1e-7)
        specificity = self.tn / (self.tn + self.fp + 1e-7)
        return (recall + specificity) / 2.0

    def reset_state(self):
        self.tp.assign(0.0)
        self.tn.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


def focal_loss(alpha=ALPHA, gamma=GAMMA):
    """
    Binary focal loss — down-weights easy (high-confidence) examples.

    Standard cross-entropy treats all samples equally, which can let the
    majority class dominate. The modulating factor (1 - p_t)^gamma reduces
    the gradient contribution of well-classified samples so the model focuses
    on difficult / ambiguous X-rays.
    Args:
        alpha (float): Controls the down-weighting of easy examples by reducing their loss contribution
        gamma (float): Balances class importance to address class imbalance, typically assigning higher weight to the rare class
    Returns: 
        function: The loss function with ALPHA and GAMMA applied
    """
    def loss(y_true, y_pred):
        """
        Computes the loss function
        Args:
            y_true: The labels of the data
            y_pred: The predictions by the model
        Returns:
            Tensor: The loss function applied in the form of a tensor
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)     # Avoid log(0)

        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)

        return tf.reduce_mean(alpha_factor * modulating_factor * bce)

    return loss


def build_model():
    """
    Construct a MobileNetV2 transfer-learning model for binary pneumonia detection.

    Architecture:
       Input → Augmentation → MobileNetV2 (frozen) → GAP → Dropout → Dense(sigmoid)

    The base is frozen initially so only the classification head is trained.
    fine_tune_model() later unfreezes the top 80 layers for domain adaptation.

    Light augmentation ideal for chest X-rays: flips are medically valid;
    rotation and zoom are kept small to avoid unrealistic distortions.

    Returns:
        tuple:  [0] The model (tf.keras.Model)\n
                [1] The base model with frozen neurons
    """
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.01),
            layers.RandomZoom(0.02),
        ],
        name="augmentation",
    )

    # ImageNet-pretrained base; top (classification) layers excluded
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False      # Freeze the base initially

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="input_image")
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)      # Scales to [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.40, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier")(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_pneumonia")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(INITIAL_LEARNING_RATE),
        loss=focal_loss(),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            BalancedAccuracy(),         # Primary monitor (see get_callbacks)
        ],
    )

    return model, base


def get_callbacks(stage_name="baseline"):
    """
    Callbacks for early stopping if metrics are not improving by epoch.

    Args:
        stage_name (str): Name of the file
    Returns:
        list: A list with all the callbacks for early stopping
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    return [
        # Stop early if val_balanced_accuracy stops improving for 4 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor="val_balanced_accuracy",
            patience=4,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
         # Save only the epoch with the highest val_balanced_accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f"best_{stage_name}.keras"),
            monitor="val_balanced_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


def fine_tune_model(model:tf.keras.Model, base, train_ds, val_ds, class_weight:dict):
    """
    Trains the model on the unfreeze top 80 MobileNetV2 layers and continue training at a lower LR.

    Only the deeper layers (closer to the output) are unfrozen — they encode
    higher-level features that benefit most from domain adaptation to X-ray images.
    The very early layers (texture / edge detectors) remain frozen to preserve
    general low-level representations.
    Args:
        model (tf.keras.Model): The model which will be fine tuned
        base: The frozen model layers
        train_ds: The training dataset
        val_ds: The validation dataset
        class_weight: The weight of each class
    """
    # Unfreeze the top 80 MobileNetV2 layers and continue training at a lower LR.
    print("\nStarting fine-tuning...")

    base.trainable = True
    for layer in base.layers[:-80]:         # Keep early layers frozen
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINE_TUNE_LEARNING_RATE),
        loss=focal_loss(),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            BalancedAccuracy(),
        ],
    )

    # initial_epoch=EPOCHS continues the epoch counter from where head training stopped
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=EPOCHS,
        callbacks=get_callbacks(stage_name="fine_tuned"),
        class_weight=class_weight,
        verbose=1,
    )


def compute_roc_points(labels, probs):
    """
    Manually compute TPR/FPR pairs across all unique probability thresholds.
    Args:
        labels (list): The labels of the dataset
        probs (list): The model's prediction probability
    Returns:
        tuple:  [0] FalsePositiveRates\n
                [1] TruePositiveRates\n
                [2] Computed AUC
    """
    # 
    thresholds = np.unique(np.concatenate(([0.0], probs, [1.0])))
    thresholds = np.sort(thresholds)[::-1]

    fprs = []
    tprs = []

    for t in thresholds:
        preds = (probs >= t).astype(int)

        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        tpr = tp / (tp + fn + 1e-7)
        fpr = fp / (fp + tn + 1e-7)

        tprs.append(tpr)
        fprs.append(fpr)

    fprs = np.array(fprs)
    tprs = np.array(tprs)
    order = np.argsort(fprs)
    fprs = fprs[order]
    tprs = tprs[order]

    auc = np.trapz(tprs, fprs)
    return fprs, tprs, float(auc)

def find_best_threshold(model, val_ds, preferred_threshold=0.745, tolerance=0.002):
    """
    Select the decision threshold on the validation set.

    The search range is intentionally biased toward 0.745 because a higher threshold
    reduces false positives (flagging healthy patients as sick) at the cost of
    slightly lower recall. The tolerance parameter allows a nearby threshold to
    win over a marginally better one if it is closer to the preferred value,
    providing stability across training runs.

    Args:
        model: The model for to find the threshold
        val_ds: Validation Dataset
        preferred_threshold (float): 
        tolerance (float):
    Returns:
        tuple: best_threshold, best_bal_acc, sweep, labels, probs, fprs, tprs, roc_auc
    """
    probs = model.predict(val_ds, verbose=1).flatten()

    labels = []
    for _, y in val_ds:
        labels.extend(y.numpy().flatten())
    labels = np.array(labels).astype(int)

    thresholds = np.round(np.arange(0.50, 0.851, 0.005), 3)

    best_threshold = 0.5
    best_bal_acc = -1.0
    best_gap = float("inf")
    sweep = []

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)

        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        recall = tp / (tp + fn + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        bal_acc = (recall + specificity) / 2.0

        sweep.append(
            {
                "threshold": float(threshold),
                "balanced_accuracy": float(bal_acc),
                "recall": float(recall),
                "specificity": float(specificity),
                "precision": float(precision),
            }
        )

        gap = abs(float(threshold) - preferred_threshold)

        if bal_acc > best_bal_acc + tolerance:
            best_bal_acc = bal_acc
            best_threshold = float(threshold)
            best_gap = gap
        elif abs(bal_acc - best_bal_acc) <= tolerance and gap < best_gap:
            best_threshold = float(threshold)
            best_gap = gap

    fprs, tprs, roc_auc = compute_roc_points(labels, probs)

    print(f"\nBest validation threshold: {best_threshold:.3f}")
    print(f"Best validation balanced accuracy: {best_bal_acc:.4f}")
    print(f"Validation ROC AUC: {roc_auc:.4f}")

    return best_threshold, best_bal_acc, sweep, labels, probs, fprs, tprs, roc_auc

# def find_best_threshold(model, val_ds):
#     probs = model.predict(val_ds, verbose=1).flatten()
#
#     labels = []
#     for _, y in val_ds:
#         labels.extend(y.numpy().flatten())
#     labels = np.array(labels).astype(int)
#
#     thresholds = np.linspace(0.10, 0.90, 161)
#
#     best_threshold = 0.5
#     best_bal_acc = -1.0
#     sweep = []
#
#     for threshold in thresholds:
#         preds = (probs >= threshold).astype(int)
#
#         tp = ((preds == 1) & (labels == 1)).sum()
#         tn = ((preds == 0) & (labels == 0)).sum()
#         fp = ((preds == 1) & (labels == 0)).sum()
#         fn = ((preds == 0) & (labels == 1)).sum()
#
#         recall = tp / (tp + fn + 1e-7)
#         specificity = tn / (tn + fp + 1e-7)
#         precision = tp / (tp + fp + 1e-7)
#         bal_acc = (recall + specificity) / 2.0
#
#         sweep.append(
#             {
#                 "threshold": float(threshold),
#                 "balanced_accuracy": float(bal_acc),
#                 "recall": float(recall),
#                 "specificity": float(specificity),
#                 "precision": float(precision),
#             }
#         )
#
#         if bal_acc > best_bal_acc:
#             best_bal_acc = bal_acc
#             best_threshold = float(threshold)
#
#     fprs, tprs, roc_auc = compute_roc_points(labels, probs)
#
#     print(f"\nBest validation threshold: {best_threshold:.2f}")
#     print(f"Best validation balanced accuracy: {best_bal_acc:.4f}")
#     print(f"Validation ROC AUC: {roc_auc:.4f}")
#
#     return best_threshold, best_bal_acc, sweep, labels, probs, fprs, tprs, roc_auc


def save_threshold(threshold, balanced_accuracy):
    """
    Saves the threshold to THRESHOLD_FILE location
    Args:
        threshold (float): Threshold to be saved
        balanced_accuracy (float): Accuracy to be saved
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(THRESHOLD_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": threshold,
                "validation_balanced_accuracy": balanced_accuracy,
            },
            f,
            indent=2,
        )
    print(f"Saved threshold info to: {THRESHOLD_FILE}")


def save_threshold_sweep(sweep):
    """
    Saves the sweep to the THRESHOLD_SWEEP_FILE location
    Args:
        sweep: The sweep to be saved
    """
    with open(THRESHOLD_SWEEP_FILE, "w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2)
    print(f"Saved threshold sweep to: {THRESHOLD_SWEEP_FILE}")


def save_roc_data(labels, probs, fprs, tprs, roc_auc):
    """
    Saves the ROC data into the ROC_FILE file
    Args:
        labels: Dataset labels
        probs: The models predicted probabilities
        fprs: False Positive Rates
        tprs: True Postive Rates
        roc_auc (float): The calculated ROC_AUC value
    """
    np.savez(
        ROC_FILE,
        labels=labels,
        probs=probs,
        fprs=fprs,
        tprs=tprs,
        roc_auc=roc_auc,
    )
    print(f"Saved ROC data to: {ROC_FILE}")


def main():
    # Data preparation: stratified split, dataset pipelines, class weights
    train_paths, train_labels, val_paths, val_labels = stratified_split()

    inspect_labels(train_labels, name="Training")
    inspect_labels(val_labels, name="Validation")

    train_ds = make_dataset_from_paths(train_paths, train_labels, shuffle=True)
    val_ds = make_dataset_from_paths(val_paths, val_labels, shuffle=False)
    test_ds, class_names = build_test_dataset()

    print("\nClass names:", class_names)

    # Softer manual weights than the previous aggressive automatic weighting.
    # NORMAL gets a higher weight (1.50) to compensate for the dataset imbalance
    # (~1,341 NORMAL vs ~3,875 PNEUMONIA in the training fold).
    class_weight = {
        0: 1.50,  # NORMAL — upweighted due to fewer samples
        1: 0.85,  # PNEUMONIA — slightly downweighted
    }

    print("Using class weights:", class_weight)

    # Initial stage: Model construction and head training
    model, base = build_model()
    model.summary()

    print("\nStarting baseline training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks(stage_name="baseline"),
        class_weight=class_weight,
        verbose=1,
    )

    # Fine-tuning stage: Unfreeze top layers and continue training with a lower learning rate
    fine_tune_model(model, base, train_ds, val_ds, class_weight)

    # Threshold selection, saving and ROC data preparation
    best_threshold, best_bal_acc, sweep, labels, probs, fprs, tprs, roc_auc = find_best_threshold(model, val_ds)
    save_threshold(best_threshold, best_bal_acc)
    save_threshold_sweep(sweep)
    save_roc_data(labels, probs, fprs, tprs, roc_auc)

    # Save the final model (architecture + weights) for later inference and evaluation on the test set
    os.makedirs(MODELS_DIR, exist_ok=True)
    #model.export(BASELINE_MODEL_DIR)
    # Using tf.saved_model.save with experimental_custom_gradients=False
    # to avoid ReplicaContext error when saving after fine-tuning
    tf.saved_model.save(
        model,
        BASELINE_MODEL_DIR,
        options=tf.saved_model.SaveOptions(experimental_custom_gradients=False)
    )
    model.save(os.path.join(MODELS_DIR, "baseline.keras"))
    model.save_weights(os.path.join(MODELS_DIR, "baseline.weights.h5"))

    print(f"\nSaved baseline model to: {BASELINE_MODEL_DIR}")

    # Final evaluation on the test set at default 0.5 threshold by Keras
    results = model.evaluate(test_ds, verbose=1)
    print("Test results:", dict(zip(model.metrics_names, results)))


if __name__ == "__main__":
    main()
