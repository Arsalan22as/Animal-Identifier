import tensorflow as tf
import numpy as np
import os

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

def create_and_train_model():
    try:
        # Create a simple sequential model with proper input layer
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        # Generate some dummy data for training
        X_train = np.random.random((1000, 10))
        y_train = np.random.random((1000, 1))

        # Train the model
        print("Training model...")
        history = model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

        # Save the model
        model_path = 'saved_model.keras'
        model.save(model_path)
        print(f"Model successfully saved to '{model_path}'")

        # Verify the model can be loaded
        loaded_model = tf.keras.models.load_model(model_path)
        print("Model successfully loaded and verified")

        return model, history

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

if __name__ == "__main__":
    model, history = create_and_train_model() 