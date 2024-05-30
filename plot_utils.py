import matplotlib.pyplot as plt
import numpy as np

def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_predictions(predictions, y_test, num_samples=10):
    max_pixel_value = 255
    y_test_adj = y_test * max_pixel_value
    predictions_adj = predictions * max_pixel_value

    y_test_adj_avg = np.mean(y_test_adj[:, :, :, :3], axis=-1)
    predictions_adj_avg = np.mean(predictions_adj[:, :, :, :3], axis=-1)

    # Randomly select num_samples indices
    indices = np.random.choice(len(y_test_adj), num_samples, replace=False)

    fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    for i, idx in enumerate(indices):
        axs[0, i].imshow(y_test_adj_avg[idx, :, :].astype(int), cmap='gray')
        axs[0, i].set_title('Ground Truth')
        axs[0, i].axis('off')
        
        axs[1, i].imshow(predictions_adj_avg[idx, :, :].astype(int), cmap='gray')
        axs[1, i].set_title('Prediction')
        axs[1, i].axis('off')
        
    plt.tight_layout()
    plt.show()
