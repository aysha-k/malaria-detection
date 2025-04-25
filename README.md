# Malaria Detection Using CNN

This repository contains a Jupyter Notebook for detecting malaria in blood cell images using a Convolutional Neural Network (CNN). The model is trained to classify images into two categories: "Parasitized" (infected) and "Uninfected" (healthy).

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Overview
The project aims to automate the detection of malaria in blood cell images using deep learning. The CNN model is built using TensorFlow and Keras, and it achieves reasonable accuracy in classifying infected and uninfected cells.

## Dataset
The dataset used in this project is organized into two main folders:
- `train`: Contains training images (354 images)
- `test`: Contains test images (134 images)

Each folder has subdirectories for the two classes:
- `Parasitized`: Images of infected cells
- `Uninfected`: Images of healthy cells

## Model Architecture
The CNN model consists of the following layers:
1. Convolutional Layer (32 filters, 3x3 kernel, ReLU activation)
2. Max Pooling Layer (2x2 pool size)
3. Dropout Layer (25%)
4. Convolutional Layer (64 filters, 3x3 kernel, ReLU activation)
5. Max Pooling Layer (2x2 pool size)
6. Dropout Layer (25%)
7. Convolutional Layer (128 filters, 3x3 kernel, ReLU activation)
8. Max Pooling Layer (2x2 pool size)
9. Dropout Layer (25%)
10. Flatten Layer
11. Dense Layer (128 neurons, ReLU activation)
12. Dropout Layer (50%)
13. Output Layer (1 neuron, sigmoid activation)

The model is compiled with the Adam optimizer and binary cross-entropy loss function.

## Training
The model was trained for 20 epochs with the following parameters:
- Batch size: 32
- Image dimensions: 128x128 pixels
- Data augmentation (rotation, width/height shift, horizontal flip, zoom)

## Evaluation
The model achieved the following performance metrics:
- **Test Accuracy**: 54%
- **Classification Report**:
  ```
              precision    recall  f1-score   support
  Uninfected       0.65      0.69      0.67        91
  Parasitized       0.24      0.21      0.23        43
    accuracy                           0.54       134
   macro avg       0.45      0.45      0.45       134
  weighted avg       0.52      0.54      0.53       134
  ```
- **Confusion Matrix**:
  ```
  [[63, 28],
   [34,  9]]
  ```

## Results
The training and validation accuracy/loss curves are plotted in the notebook, showing the model's learning progress over epochs. The model is saved as `malaria_cnn_model.h5` for future use.

## Usage
1. Clone this repository
2. Install the required dependencies (see below)
3. Run the Jupyter Notebook `malaria_detection.ipynb`
4. The notebook includes all steps from data loading to model evaluation

## Dependencies
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-learn
- OpenCV (for image processing)

Install dependencies using:
```bash
pip install tensorflow numpy matplotlib scikit-learn opencv-python
```

## License
This project is open-source and available under the [MIT License](LICENSE).

## Note
The current model accuracy is moderate (54%). Future improvements could include:
- Using a larger dataset
- Trying more complex architectures
- Hyperparameter tuning
- Transfer learning with pre-trained models

Feel free to contribute to this project by submitting issues or pull requests!
