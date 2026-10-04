from pathlib import Path

import numpy as np
import yaml
from sklearn .model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def main():
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as file:
        params = yaml.safe_load(file)["preprocess"]

    x_train = np.load(RAW_DIR / "x_train.npy")
    y_train = np.load(RAW_DIR / "y_train.npy")
    x_test = np.load(RAW_DIR / "x_test.npy")
    y_test = np.load(RAW_DIR / "y_test.npy")

    x_train = x_train.astype("float32") / 256.0
    x_test = x_test.astype("float32") / 256.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=y_train,
    )


    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    arrays = {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
        }

    for name, array in arrays.items():
        np.save(PROCESSED_DIR / f"{name}.npy", array)

    print(f"Training images: {x_train.shape}")
    print(f"Validation images: {x_val.shape}")
    print(f"Test images: {x_test.shape}")
    print(f"Processed image range: {x_train.min()} to {x_train.max()}")


if __name__ == "__main__":
    main()


# Temporary note: validation is split from the training data.