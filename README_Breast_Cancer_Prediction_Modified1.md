# Breast Cancer Prediction Using CNN on Histopathology Images

## Overview
This project implements a Convolutional Neural Network (CNN) model to predict breast cancer from histopathology images. The dataset used includes images from the CBIS-DDSM breast cancer image dataset and breast histopathology images dataset. The notebook performs data preprocessing, exploratory data analysis, model building, training, evaluation, and visualization of results.

## Dataset
- CBIS-DDSM breast cancer image dataset (includes cropped images, full mammogram images, and ROI mask images).
- Breast histopathology images dataset with labeled cancer and non-cancer images.

## Data Preprocessing
- Images are loaded and resized to 50x50 pixels.
- Labels are binary: 0 for no cancer, 1 for cancer.
- Data is shuffled and split into training (75%) and testing (25%) sets.
- Labels are one-hot encoded for binary classification.

## Model Architecture
- CNN with 4 convolutional layers:
  - Conv2D with 32 filters, kernel size (3,3), ReLU activation, padding='same'
  - MaxPooling2D with stride 2
  - Conv2D with 64 filters, kernel size (3,3), ReLU activation, padding='same'
  - MaxPooling2D with pool size (3,3), stride 2
  - Conv2D with 128 filters, kernel size (3,3), ReLU activation, padding='same'
  - MaxPooling2D with pool size (3,3), stride 2
  - Conv2D with 128 filters, kernel size (3,3), ReLU activation, padding='same'
  - MaxPooling2D with pool size (3,3), stride 2
- Flatten layer
- Dense layer with 128 units, ReLU activation
- Output Dense layer with 2 units, Softmax activation

## Training
- Optimizer: Adam with learning rate 0.001
- Loss function: Binary crossentropy
- Metrics: Accuracy
- Epochs: 25
- Batch size: 75

## Results
- The model was evaluated on the test set.
- Accuracy and loss curves were plotted for both training and validation sets.
- Confusion matrix was generated to visualize classification performance.
- Example predictions on test images were shown.

## Accuracy
- The model achieves approximately 90% accuracy on the test set based on the evaluation and accuracy plots in the notebook.

## Modifications for Current Environment
- The original Kaggle-specific data import and symlink creation code has been removed to allow the notebook to run in a non-Kaggle environment.
- Users need to ensure the datasets are available locally and update the data paths accordingly.

## Usage
- Run the modified notebook to reproduce the results.
- Ensure the datasets are available in the specified input paths or update paths as needed.
- Modify parameters such as image size, batch size, and epochs as needed.

## Dataset Instructions
- The datasets used in this project are large and not included in the repository.
- You can download the following datasets from Kaggle:
  - CBIS-DDSM breast cancer image dataset: https://www.kaggle.com/datasets/cbiskaggle/cbis-ddsm-breast-cancer-image-dataset
  - Breast histopathology images dataset: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
- After downloading, place the datasets in a local directory.
- Update the data paths in the notebook accordingly to point to your local dataset directories.

## Dependencies
- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Seaborn
- Plotly

## Author
This notebook was created for breast cancer detection using deep learning techniques on histopathology images.
