import os
import argparse
import tensorflow as tf
from data_utils import download_and_extract_data, load_and_preprocess_images, generate_low_res_hsi_and_high_res_rgb, normalize_data
from model_utils import create_model
from plot_utils import plot_history, plot_predictions
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

def main(model_path, batch_size, epochs, learning_rate, filters, blocks, save_every):
    # Check if TensorFlow is using GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPUs available: {len(physical_devices)}")
        for device in physical_devices:
            print(f"Device: {device}")
    else:
        print("No GPU available, using CPU instead.")

    data_url = 'https://www1.cs.columbia.edu/CAVE/databases/multispectral/zip/complete_ms_data.zip'
    root_folder = 'complete_ms_data'
    target_size = 64

    # Download and extract data
    download_and_extract_data(data_url, root_folder)

    # Load and preprocess images
    cropped_images = load_and_preprocess_images(root_folder, target_size)
    LowResHSI, HiResRGB = generate_low_res_hsi_and_high_res_rgb(cropped_images)

    y = cropped_images
    X_low_res_hsi = LowResHSI
    X_hi_res_rgb = HiResRGB

    X_low_res_hsi_train, X_low_res_hsi_test, y_train, y_test = train_test_split(X_low_res_hsi, y, test_size=0.2, random_state=42)
    X_hi_res_rgb_train, X_hi_res_rgb_test = train_test_split(X_hi_res_rgb, test_size=0.2, random_state=42)

    X_low_res_hsi_train, X_low_res_hsi_test, X_hi_res_rgb_train, X_hi_res_rgb_test, y_train, y_test = normalize_data(
        X_low_res_hsi_train, X_low_res_hsi_test, X_hi_res_rgb_train, X_hi_res_rgb_test, y_train, y_test)

    history = None  # Initialize history variable

    if model_path and os.path.exists(model_path):
        try:
            print(f"Loading model from {model_path}...")
            model = load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training a new model instead...")
            model, history = create_and_train_model(X_hi_res_rgb_train, X_low_res_hsi_train, y_train, X_hi_res_rgb_test, X_low_res_hsi_test, y_test, model_path, batch_size, epochs, learning_rate, filters, blocks, save_every)
    else:
        print("Training new model...")
        model, history = create_and_train_model(X_hi_res_rgb_train, X_low_res_hsi_train, y_train, X_hi_res_rgb_test, X_low_res_hsi_test, y_test, model_path, batch_size, epochs, learning_rate, filters, blocks, save_every)

    # Evaluate the model on the testing set
    loss, accuracy = model.evaluate([X_hi_res_rgb_test, X_low_res_hsi_test], y_test)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

    # Make predictions and plot them
    predictions = model.predict([X_hi_res_rgb_test, X_low_res_hsi_test])
    plot_predictions(predictions, y_test)

    # Plot training history if available
    if history:
        plot_history(history)

def create_and_train_model(X_hi_res_rgb_train, X_low_res_hsi_train, y_train, X_hi_res_rgb_test, X_low_res_hsi_test, y_test, model_path, batch_size, epochs, learning_rate, filters, blocks, save_every):
    model = create_model(filters, blocks, learning_rate)
    model.summary()

    callbacks = []
    if save_every:
        steps_per_epoch = len(X_hi_res_rgb_train) // batch_size
        save_freq = steps_per_epoch * save_every
        checkpoint_path = model_path.replace('.h5', '_epoch_{epoch:02d}.h5')
        checkpoint_callback = ModelCheckpoint(checkpoint_path, save_weights_only=False, save_freq=save_freq)
        callbacks.append(checkpoint_callback)

    history = model.fit(
        [X_hi_res_rgb_train, X_low_res_hsi_train],
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([X_hi_res_rgb_test, X_low_res_hsi_test], y_test),
        callbacks=callbacks
    )

    if not save_every:
        model.save(model_path)
        print(f"Model saved to {model_path}")

    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or load a superresolution model.')
    parser.add_argument('--model_path', type=str, default='my_model.h5', help='Path to the model file to load.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer.')
    parser.add_argument('--filters', type=int, default=64, help='Number of filters for the convolutional layers.')
    parser.add_argument('--blocks', type=int, default=3, help='Number of residual blocks in the encoder and decoder.')
    parser.add_argument('--save_every', type=int, default=0, help='Save the model every specified number of epochs. If 0
