# 1-D and 2-D Convolutional Neural Networks in PyTorch

This repository contains an implementation of 1-Dimensional (1D) and 2-Dimensional (2D) Convolutional Neural Networks (CNNs) using PyTorch. The implementation is designed to be flexible, allowing the user to easily create and experiment with different CNN architectures. The main classes provided are `CNN1D` and `CNN2D`, which can be used for various tasks such as image classification, audio processing, or time-series analysis.

## Overview

This project demonstrates the application of 1D and 2D CNNs on different datasets. It begins with the implementation of a 2D CNN for image classification on the MNIST dataset, which consists of hand-drawn digits. The `CNN2D` class is used for this purpose, and its performance is evaluated on the dataset.

Next, the implementation is extended to handle 1D input samples by defining a new class, `CNN1D`, which inherits from `CNN2D`. This 1D CNN is then applied to an Electrocardiogram (ECG) dataset for classification into "normal" or "arrhythmia" classes.

## Usage

To use the provided classes, first import the required packages:

```python
import numpy as np
import torch
import pandas
import matplotlib.pyplot as plt
```

Then, create an instance of the `CNN2D` or `CNN1D` class with the desired parameters:

```python
model = CNN2D(n_inputs, n_hiddens_per_conv_layer, n_hiddens_per_fc_layer, n_outputs,
              patch_size_per_conv_layer, stride_per_conv_layer, activation_function='tanh', device='cpu')
```

For the 1D CNN, use:

```python
model = CNN1D(n_inputs, n_hiddens_per_conv_layer, n_hiddens_per_fc_layer, n_outputs,
              patch_size_per_conv_layer, stride_per_conv_layer, activation_function='tanh', device='cpu')
```

After creating the model, train it on your dataset using the provided `train` method:

```python
model.train(X_train, T_train, n_epochs, learning_rate)
```

Finally, test the model's performance on a test dataset using the `use` method:

```python
predicted_labels, _ = model.use(X_test)
```

## Example

The notebook provided in this repository demonstrates the usage of the 1D and 2D CNNs on the MNIST dataset and an ECG dataset. The notebook is organized as an experiment, guiding the viewer through the creation and application of various CNN architectures.

## Requirements

To run the code, you will need the following packages:

- NumPy
- PyTorch
- Pandas
- Matplotlib

Ensure that you have the latest version of these packages installed in your environment.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
