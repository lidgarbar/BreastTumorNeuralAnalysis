# BreastTumorNeuralAnalysis

This document provides an overview of two Python scripts: one for creating and training a neural network model and the other for data preprocessing. Both scripts are meant for machine learning tasks.

## Neural Network Model

### Code Overview

- Import libraries, load data, and preprocess it.
- Create a class `NeuralModel` to define, train, evaluate, and test a neural network.
- Build a neural network model using Keras.
- Train the model with early stopping to prevent overfitting.
- Evaluate and test the model's performance.
- Plot the training history.

### Usage

1. Import your data or modify the data loading section.
2. Customize the model architecture and training parameters.
3. Run the script to create, train, and evaluate the model.
4. Visualize the training history and assess the model's performance.

## Data Preprocessing

### Code Overview

- Load data from a CSV file, normalize it, and split it into training, validation, and test sets.
- Define a class-based approach for data normalization.
- Print statistics about the data normalization.
- Use scikit-learn to split the data into sets.

### Usage

1. Update `csv_path` to point to your CSV file.
2. Customize data transformations and preprocessing.
3. Run the script to load, preprocess, and split your data.
4. Modify print statements or add additional processing steps if needed.

## Dependencies

- numpy
- pandas
- matplotlib
- scikit-learn

## Authors

- Juan Sánchez Moreno @juansm01
- Lidia García Barragan @lidgarbar

Please make sure to credit the original author and provide any necessary citations if you use or modify this code in your projects.
