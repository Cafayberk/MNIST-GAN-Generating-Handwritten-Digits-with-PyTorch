#MNIST GAN: Generating Handwritten Digits with PyTorch

This project implements a Generative Adversarial Network (GAN) using PyTorch to generate synthetic handwritten digits based on the MNIST dataset. The GAN consists of a Generator that produces fake images from random noise and a Discriminator that distinguishes between real and fake images. Through adversarial training, the Generator learns to produce images that closely resemble real handwritten digits.

Features

PyTorch-based implementation of a simple fully connected GAN.
Uses the MNIST dataset for training.
Customizable hyperparameters such as learning rates, noise vector size, and training epochs.
Generates sample images after training.
Compatible with both CPU and GPU execution.

Requirements

Python 3.8+
PyTorch
Torchvision
Matplotlib
NumPy

You can install the dependencies with: pip install torch torchvision matplotlib numpy

How It Works

Data Loading: MNIST dataset is loaded and normalized between -1 and 1.
Discriminator: A feedforward network that outputs a probability indicating whether an input image is real or fake.
Generator: Takes a random noise vector and outputs a 28x28 image.
Training: The Discriminator is trained to correctly classify real and fake images.
          The Generator is trained to fool the Discriminator into classifying fake images as real.
Image Generation: After training, the Generator produces new handwritten digit images from random noise.

Usage

Clone this repository: git clone https://github.com/yourusername/mnist-gan-pytorch.git
Open the notebook or script in Google Colab or locally.
Run the training script: python gan_mnist.py
After training, generated images will be displayed using Matplotlib.

Hyperparameters

Batch Size: 128
Image Size: 28 × 28
Noise Vector Size (z_dim): 100
Learning Rate (Generator): 0.0002
Learning Rate (Discriminator): 0.0001
Epochs: 50

License
This project is released under the MIT License.
