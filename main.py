from data_utils import download_and_extract_data, load_and_preprocess_images, generate_low_res_hsi_and_high_res_rgb, normalize_data
from model_utils import create_model
from plot_utils import plot_history, plot_predictions
from sklearn.model_selection import train_test_split

def main():
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

    model = create_model()
    model.summary()

    history = model.fit(
        [X_hi_res_rgb_train, X_low_res_hsi_train],
        y_train,
        batch_size=32,
        epochs=10,
        validation_data=([X_hi_res_rgb_test, X_low_res_hsi_test], y_test)
    )

    loss, accuracy = model.evaluate([X_hi_res_rgb_test, X_low_res_hsi_test], y_test)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)
    model.save("my_model.h5")

    plot_history(history)

    predictions = model.predict([X_hi_res_rgb_test, X_low_res_hsi_test])
    plot_predictions(predictions, y_test)

if __name__ == "__main__":
    main()
