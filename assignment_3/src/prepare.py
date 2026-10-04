from pathlib import Path
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def main():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    np.save(RAW_DIR / "x_train.npy", x_train)
    np.save(RAW_DIR / "y_train.npy", y_train)
    np.save(RAW_DIR / "x_test.npy", x_test)
    np.save(RAW_DIR / "y_test.npy", y_test)

    print("Raw Fashion-MNIST data saved.")
    print("Training images:", x_train.shape)
    print("Testing images:", x_test.shape)


if __name__ == "__main__":
    main()

# Temporary note: validation is split from the training data.
# Raw arrays preserve the original dataset values.