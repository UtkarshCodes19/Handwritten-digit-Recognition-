import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

MODEL_PATH = "digit_model.keras"

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixels from [0,255] to [0,1] and add channel dimension.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    model = build_model()
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    plt.figure()
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_accuracy.png", dpi=150)
    plt.close()

    # Show a few predictions in a saved figure.
    predictions = model.predict(x_test[:10], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(10, 4))
    for i in range(10):
        ax = plt.subplot(2, 5, i + 1)
        ax.imshow(x_test[i].squeeze(), cmap="gray")
        ax.set_title(f"True: {y_test[i]} | Pred: {predicted_labels[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
