import os
import argparse
import tensorflow as tf
from data_utils import download_and_extract_data, load_and_preprocess_images, generate_low_res_hsi_and_high_res_rgb, normalize_data
from model_utils import create_model
from plot_utils import plot_history, plot_predictions
from sklearn.model_selection import train_test_split
from keras.models import load_model

def main(model_path=None):
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

    if model_path and os.path.exists(model_path):
        try:
            print(f"Loading model from {model_path}...")
            model = load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training a new model instead...")
            model = create_and_train_model(X_hi_res_rgb_train, X_low_res_hsi_train, y_train, X_hi_res_rgb_test, X_low_res_hsi_test, y_test, model_path)
    else:
        print("Training new model...")
        model = create_and_train_model(X_hi_res_rgb_train, X_low_res_hsi_train, y_train, X_hi_res_rgb_test, X_low_res_hsi_test, y_test, model_path)

    # Evaluate the model on the testing set
    loss, accuracy = model.evaluate([X_hi_res_rgb_test, X_low_res_hsi_test], y_test)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

    # Make predictions and plot them
    predictions = model.predict([X_hi_res_rgb_test, X_low_res_hsi_test])
    plot_predictions(predictions, y_test)

def create_and_train_model(X_hi_res_rgb_train, X_low_res_hsi_train, y_train, X_hi_res_rgb_test, X_low_res_hsi_test, y_test, model_path):
    model = create_model()
    model.summary()

    history = model.fit(
        [X_hi_res_rgb_train, X_low_res_hsi_train],
        y_train,
        batch_size=32,
        epochs=10,
        validation_data=([X_hi_res_rgb_test, X_low_res_hsi_test], y_test)
    )

    if model_path:
        model.save(model_path)
        print(f"Model saved to {model_path}")

    plot_history(history)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or load a superresolution model.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model file to load.')
    
    args = parser.parse_args()
    main(model_path=args.model_path)
