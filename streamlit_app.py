import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the MNIST dataset
@st.cache(allow_output_mutation=True)
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data
    return (x_train, y_train), (x_test, y_test)

# Build a simple neural network model
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Plot training history
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='accuracy')
    ax1.plot(history.history['val_accuracy'], label='val_accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.set_title('Training and Validation Accuracy')

    ax2.plot(history.history['loss'], label='loss')
    ax2.plot(history.history['val_loss'], label='val_loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.set_title('Training and Validation Loss')
    st.pyplot(fig)

st.title("MNIST Digit Classifier Training")

st.write("This app trains a simple neural network on the MNIST dataset.")

(x_train, y_train), (x_test, y_test) = load_data()

# Show a sample of the training data
st.write("### Sample Training Data")
fig, ax = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    ax[i].imshow(x_train[i], cmap='gray')
    ax[i].axis('off')
st.pyplot(fig)

if st.button('Train Model'):
    st.write("Training the model...")
    model = create_model()
    history = model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=2)
    st.write("Training completed.")
    st.write("### Training History")
    plot_history(history)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    st.write(f"Test accuracy: {test_acc}")

