# Hybrid_Methods

This project contains two main components: leNet and Gan. Below is a brief description of each:

## leNet

The leNet function implements a Convolutional Neural Network (CNN) using the LeNet architecture for image classification. It is designed to work with a dataset of handwritten digits. Here are the main steps performed by leNet:

Data Import and Preprocessing:
Reads a dataset from 'data/numbers.csv' and removes unnecessary columns.
Imports training and testing image data from IDX files.
Normalizes pixel values and pads images.
Model Design:
Constructs a LeNet-style CNN model using TensorFlow/Keras.
Defines parameters such as input shape, filter sizes, and the number of classes.
Training:
Compiles and trains the CNN model on the training data.
Displays training images and performance curves (accuracy, precision) if plotting is enabled.
Validation:
Validates the trained model on a separate set of handwritten digits.
Displays predicted classes and confusion matrix if plotting is enabled.

## Gan

The Gan function implements a simple Generative Adversarial Network (GAN) for image generation. Here is an overview of the Gan function:

Data Import and Preprocessing:
Imports training and testing image data from IDX files.
Reshapes and normalizes pixel values.
Model Design:
Constructs a GAN model with a generator and a discriminator using TensorFlow/Keras.
Defines generator and discriminator architectures.
Training:
Trains the GAN model using the specified optimizer and loss functions.
Displays reconstruction results and loss curves if plotting is enabled.

## Running the Code

To run the code, execute the script with Python. Make sure to activate the virtual environment and install the required packages using the provided requirements.txt file.

```
# Activate virtual environment
source venv/bin/activate  # for Linux/Mac
venv\Scripts\activate      # for Windows

# Install required packages
pip install -r requirements.txt

# Run the main script
python main_script.py
```
The code includes visualization components that display training and validation results if plotting is set to True in both leNet and Gan.

Note: The code assumes a specific directory structure and file naming conventions for the dataset and may need adjustments based on your specific setup. Ensure that the required dataset files are present in the specified locations.
