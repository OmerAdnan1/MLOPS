import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def main():
    x_test = np.load(PROCESSED_DIR / "x_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    model = tf.keras.models.load_model(
        PROJECT_ROOT / "models" / "model.h5",
        compile=False,
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(10)),
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )
    display.plot(
        ax=axis,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
    )
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "test_samples": int(len(y_test)),
    }

    content = json.dumps(metrics, indent=2) + "\n"

    (PROJECT_ROOT / "metrics.json").write_text(
        content,
        encoding="utf-8",
    )
    (REPORTS_DIR / "metrics.json").write_text(
        content,
        encoding="utf-8",
    )

    print(content)
    print("Saved reports/confusion_matrix.png")

    if accuracy < 0.85:
        print("Accuracy is below the assignment target of 85%.")


if __name__ == "__main__":
    main()