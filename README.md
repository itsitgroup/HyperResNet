
# SuperResolution Using Machine Learning

This project aims to achieve superresolution using machine learning models. Superresolution is a technique to enhance image resolution by combining information from multiple images of the same scene. In this project, we use RGB images with high spatial resolution and hyperspectral images with high spectral resolution to create images with both high spatial and spectral resolution.

## Dataset

The dataset used in this project is the CAVE multispectral image dataset by Columbia University. The dataset can be downloaded from [this link](https://www1.cs.columbia.edu/CAVE/databases/multispectral/zip/complete_ms_data.zip).

## Project Structure

- `data_utils.py`: Functions related to data downloading, loading, and preprocessing.
- `model_utils.py`: Functions related to model creation and training.
- `plot_utils.py`: Functions related to plotting and visualizations.
- `main.py`: The main script that orchestrates the workflow.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/superresolution-ml.git
   cd superresolution-ml
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the `main.py` script to download the dataset, preprocess the data, train the model, and visualize the results:
   ```bash
   python main.py
   ```

## Functions

### Data Utils

- `download_and_extract_data(url, dest_folder, zip_name='complete_ms_data.zip')`: Downloads and extracts the dataset.
- `load_and_preprocess_images(root_folder, target_size)`: Loads and preprocesses images.
- `generate_low_res_hsi_and_high_res_rgb(cropped_images)`: Generates low-resolution HSI and high-resolution RGB images.
- `normalize_data(...)`: Normalizes the data.

### Model Utils

- `create_model()`: Creates the superresolution model using residual and attention blocks.

### Plot Utils

- `plot_history(history)`: Plots training and validation loss.
- `plot_predictions(predictions, y_test)`: Plots ground truth and predicted images.

## Results

The model will output high-resolution hyperspectral images. Training and validation loss, as well as accuracy, will be plotted. Predictions will be compared to the ground truth images.


### requirements.txt

```plaintext
numpy
opencv-python
keras
tensorflow
matplotlib
```

### Directory Structure

Your project directory should look like this:

```
superresolution-ml/
│
├── data_utils.py
├── model_utils.py
├── plot_utils.py
├── main.py
├── README.md
└── requirements.txt
```
