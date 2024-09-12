# Cancer Detection using Neural Networks

This repository contains code for detecting cancer using a simple neural network built with TensorFlow and Keras. The dataset used in this project includes various features extracted from cell nuclei images. The goal is to predict whether a tumor is malignant (cancerous) or benign based on these features.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)

## Overview
Cancer detection is one of the critical applications in healthcare. This project demonstrates how to train a neural network to classify tumors as malignant (1) or benign (0) based on specific cell measurements. The dataset is split into training and testing sets to evaluate the model's performance. 

The neural network consists of three dense layers and uses the sigmoid activation function to output a probability indicating the likelihood of the tumor being malignant.

## Dataset
The dataset used for this project is expected to be in CSV format (`CancerStats.csv`). The dataset should contain various features for each tumor, and the target variable is the `diagnosis(1=m, 0=b)` column, where `1` represents malignant and `0` represents benign tumors.

### Columns in the dataset:
- The features are various characteristics of the tumor, such as size, texture, and smoothness.
- The target variable is the `diagnosis(1=m, 0=b)` column, where:
  - `1` stands for malignant (cancerous)
  - `0` stands for benign

## Model Architecture
The neural network model consists of three layers:
1. **Input Layer**: A dense layer with 256 units, using a sigmoid activation function.
2. **Hidden Layer**: A dense layer with 256 units, using a sigmoid activation function.
3. **Output Layer**: A dense layer with 1 unit and sigmoid activation for binary classification.

### Model Compilation:
- Optimizer: `Adam`
- Loss function: `binary_crossentropy`
- Metric: `accuracy`

### Model Summary:
```plaintext
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 256)               <calculated_param_count>
_________________________________________________________________
dense_1 (Dense)              (None, 256)               <calculated_param_count>
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 <calculated_param_count>
=================================================================
```

### Installation
To get started with this project, follow these steps:

## Clone the repository:
```
git clone https://github.com/kiruthikpurpose/CancerDetectionModel.git
```

## Install the necessary Python dependencies:

```
pip install -r Requirements.txt
```

Ensure that the dataset (CancerStats.csv) is in the root folder of the project.

### Usage

To train the model, run the following command:
```
python CancerDetectionModel.py
```

This script will:

Load the dataset from CancerStats.csv
Split it into training and testing sets
Define the neural network model
Train the model for 100 epochs
Evaluate the model on the test set

