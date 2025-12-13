import numpy as np
from sklearn.model_selection import train_test_split

FASHION_CLASSES = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',
    4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
}
NUM_CLASSES = 10
NUM_FEATURES = 784


def to_categorical(y, num_classes):
    y_encoded = np.zeros((y.size, num_classes))
    y_encoded[np.arange(y.size), y] = 1
    return y_encoded

def load_and_preprocess_data():

    print("Starting data loading and preprocessing...")

    NUM_TRAIN = 60000
    NUM_TEST = 10000
    NUM_SAMPLES = NUM_TRAIN + NUM_TEST
    np.random.seed(42)
    labels = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1), dtype=np.uint8)
    pixels = np.random.randint(0, 256, size=(NUM_SAMPLES, NUM_FEATURES), dtype=np.uint8)
    full_data = np.hstack((labels, pixels))
    # --------------------------------------------------------------------

    # Data Separation: First column is the label, the rest are features
    y_raw = full_data[:, 0].astype(np.int64)
    X_full = full_data[:, 1:].astype(np.float64)

    # 1. Normalization: Scale pixel values to [0, 1]
    X_full /= 255.0

    # 2. Label Encoding (Categorical)
    Y_encoded = to_categorical(y_raw, NUM_CLASSES)

    # --- Data Split ---
    # Split Full data into Training (70%) and Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_full, Y_encoded, test_size=0.3, random_state=42
    )

    # Split Temp (30%) into Validation (15%) and Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # 3. Transpose to (D, N) format for NN (Features x Samples)
    X_train = X_train.T
    X_val = X_val.T
    X_test = X_test.T
    y_train = y_train.T
    y_val = y_val.T
    y_test = y_test.T

    print("Data preparation complete.")

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")