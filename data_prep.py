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


#----
class NeuralNetwork:

    def __init__(self, layers, hidden_activation='relu', lr=0.01):
        self.layers = layers
        self.hidden_activation = hidden_activation
        self.lr = lr
        self.params = {} #weight bias storing
        self.memory = {} # stores values for backdrop

        self.init_weights()  #creates random weights

    def init_weights(self):
        #initialize wights randomnly  B = 0
        np.random.seed(42)

        for i in range(1, len(self.layers)):
            if self.hidden_activation == 'relu':
                self.params['W' + str(i)] = np.random.randn(
                    self.layers[i],
                    self.layers[i - 1]
                ) * np.sqrt(2.0 / self.layers[i - 1])
            else:
                self.params['W' + str(i)] = np.random.randn(
                    self.layers[i],
                    self.layers[i - 1]
                ) * np.sqrt(1.0 / self.layers[i - 1])

            # B start with 0
            self.params['b' + str(i)] = np.zeros((self.layers[i], 1))

    def sigmoid(self, z):
        #avoid overflow by clip
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def sigmoid_backward(self, a):
        return a * (1 - a)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_backward(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward_pass(self, X):
        self.memory['A0'] = X
        current_input = X
        num_layers = len(self.layers) - 1

        #pass hidden layers one by one
        for layer in range(1, num_layers):
            W = self.params['W' + str(layer)]
            b = self.params['b' + str(layer)]

            # multipy by weight and + B
            z = np.dot(W, current_input) + b
            #activ. func
            if self.hidden_activation == 'relu':
                current_input = self.relu(z)
            else:
                current_input = self.sigmoid(z)
            #save vals for backpro.
            self.memory['Z' + str(layer)] = z
            self.memory['A' + str(layer)] = current_input

        W_output = self.params['W' + str(num_layers)]
        b_output = self.params['b' + str(num_layers)]
        z_output = np.dot(W_output, current_input) + b_output
        output = self.softmax(z_output)

        self.memory['Z' + str(num_layers)] = z_output
        self.memory['A' + str(num_layers)] = output

        return output

    def calculate_loss(self, predictions, actual):
        m = actual.shape[1]
        epsilon = 1e-8
        loss = -np.sum(actual * np.log(predictions + epsilon)) / m
        return loss

    def backward_pass(self, actual_labels):
        grads = {}
        num_layers = len(self.layers) - 1
        m = actual_labels.shape[1]

        output = self.memory['A' + str(num_layers)]
        dz = output - actual_labels
        prev_activation = self.memory['A' + str(num_layers - 1)]
        grads['dW' + str(num_layers)] = np.dot(dz, prev_activation.T) / m
        grads['db' + str(num_layers)] = np.sum(dz, axis=1, keepdims=True) / m

        for layer in reversed(range(1, num_layers)):
            W_next = self.params['W' + str(layer + 1)]
            da = np.dot(W_next.T, dz)

            z = self.memory['Z' + str(layer)]
            if self.hidden_activation == 'relu':
                dz = da * self.relu_backward(z)
            else:
                a = self.memory['A' + str(layer)]
                dz = da * self.sigmoid_backward(a)

            # calc grads for specific layer
            prev_activation = self.memory['A' + str(layer - 1)]
            grads['dW' + str(layer)] = np.dot(dz, prev_activation.T) / m
            grads['db' + str(layer)] = np.sum(dz, axis=1, keepdims=True) / m

        return grads

    def update_weights(self, grads):
        num_layers = len(self.layers) - 1
        #update each layer parameters
        for layer in range(1, num_layers + 1):
            self.params['W' + str(layer)] -= self.lr * grads['dW' + str(layer)]
            self.params['b' + str(layer)] -= self.lr * grads['db' + str(layer)]

    def get_accuracy(self, predictions, true_labels):
        pred_classes = np.argmax(predictions, axis=0)  #predic class
        true_classes = np.argmax(true_labels, axis=0)  # actual class
        accuracy = np.mean(pred_classes == true_classes) * 100
        return accuracy

    def train(self, X_train, Y_train, X_val, Y_val, num_epochs=100, show_every=10):
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        print("Starting training...")
        print(f"Network structure: {self.layers}")
        print(f"Hidden layer activation: {self.hidden_activation}")
        print(f"Learning rate: {self.lr}")
        print()

        for epoch in range(num_epochs):
            output = self.forward_pass(X_train)
            loss = self.calculate_loss(output, Y_train)
            grads = self.backward_pass(Y_train)
            self.update_weights(grads)

            acc = self.get_accuracy(output, Y_train)

            val_output = self.forward_pass(X_val)
            val_loss = self.calculate_loss(val_output, Y_val)
            val_acc = self.get_accuracy(val_output, Y_val)

            train_losses.append(loss)
            train_accs.append(acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if (epoch + 1) % show_every == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - "
                      f"Loss: {loss:.4f}, Acc: {acc:.2f}% - "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        print("\nDone training!")

        results = {
            'train_loss': train_losses,
            'train_acc': train_accs,
            'val_loss': val_losses,
            'val_acc': val_accs
        }
        return results

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        output = self.forward_pass(X)
        predicted_class = np.argmax(output, axis=0)

        return predicted_class

    def test_model(self, X_test, Y_test):
        predictions = self.forward_pass(X_test)
        test_loss = self.calculate_loss(predictions, Y_test)
        test_acc = self.get_accuracy(predictions, Y_test)

        print("\n" + "=" * 50)
        print("Test Set Results:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_acc:.2f}%")
        print("=" * 50)

        return test_loss, test_acc


#----

if __name__ == '__main__':
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    print("Data loaded successfully!\n")

    print("=" * 60)
    print("NEURAL NETWORK CONFIGURATION")
    print("=" * 60)

    num_hidden = int(input("\nHow many hidden layers do you want? "))

    hidden_sizes = []
    for i in range(num_hidden):
        size = int(input(f"Number of neurons in hidden layer {i + 1}: "))
        hidden_sizes.append(size)

    activation_func = input("\nChoose activation function (relu or sigmoid): ").strip().lower()
    while activation_func not in ['relu', 'sigmoid']:
        print("Please enter either 'relu' or 'sigmoid'")
        activation_func = input("Choose activation function (relu or sigmoid): ").strip().lower()

    network_structure = [784] + hidden_sizes + [10]

    print("\n" + "=" * 60)
    print("TRAINING THE NETWORK")
    print("=" * 60 + "\n")

    model = NeuralNetwork(
        layers=network_structure,
        hidden_activation=activation_func,
        lr=0.01
    )

    history = model.train(
        X_train, y_train,
        X_val, y_val,
        num_epochs=100,
        show_every=10
    )

    model.test_model(X_test, y_test)

    print("\n" + "=" * 60)
    print("TESTING PREDICT FUNCTION")
    print("=" * 60)

    random_index = np.random.randint(0, X_test.shape[1])
    sample = X_test[:, random_index]
    true_label = np.argmax(y_test[:, random_index])

    print(f"\nActual class: {true_label} ({FASHION_CLASSES[true_label]})")

    predicted = model.predict(sample)[0]

    sample_reshaped = sample.reshape(-1, 1)
    probs = model.forward_pass(sample_reshaped)
    confidence = probs[predicted, 0] * 100

    print(f"Predicted class: {predicted} ({FASHION_CLASSES[predicted]})")
    print(f"Confidence: {confidence:.2f}%")

    if predicted == true_label:
        print("✓ Correct prediction!")
    else:
        print("✗ Wrong prediction")
