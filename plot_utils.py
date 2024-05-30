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

def plot_predictions(predictions, y_test):
    max_pixel_value = 255
    y_test_adj = y_test * max_pixel_value
    predictions_adj = predictions * max_pixel_value

    y_test_adj_avg = np.mean(y_test_adj[:, :, :, :3], axis=-1)
    predictions_adj_avg = np.mean(predictions_adj[:, :, :, :3], axis=-1)

    fig, axs = plt.subplots(8, 8, figsize=(16, 16))
    for i, ax in enumerate(axs.flat):
        ax.imshow(y_test_adj_avg[i, :, :].astype(int), cmap='gray')
        ax.set_title('Ground Truth')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(8, 8, figsize=(16, 16))
    for i, ax in enumerate(axs.flat):
        ax.imshow(predictions_adj_avg[i, :, :].astype(int), cmap='gray')
        ax.set_title('Prediction')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
