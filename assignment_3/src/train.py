import csv
import os
from pathlib import Path

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def main():
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as file:
        params = yaml.safe_load(file)["train"]

    tf.keras.utils.set_random_seed(params["seed"])
    tf.config.experimental.enable_op_determinism()

    x_train = np.load(PROCESSED_DIR / "x_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    x_val = np.load(PROCESSED_DIR / "x_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=x_train.shape[1:]),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                params["dense_units"],
                activation="relu",
            ),
            tf.keras.layers.Dropout(params["dropout_rate"]),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=params["learning_rate"]
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=2,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "model.h5")

    fieldnames = ["epoch", *history.history.keys()]

    with (MODELS_DIR / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index in range(len(history.history["loss"])):
            row = {"epoch": index + 1}
            row.update(
                {
                    name: values[index]
                    for name, values in history.history.items()
                }
            )
            writer.writerow(row)

    print("Saved models/model.h5 and models/history.csv")


if __name__ == "__main__":
    main()