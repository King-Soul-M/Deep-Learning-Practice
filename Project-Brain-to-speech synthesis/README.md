# Brain-to-speech synthesis

This project aims to create a system that can translate neural signals into comprehensible speech，Brainwave data can be converted into verbal descriptions

## Team information

Team name: ML
Team`s member and Neptun Codes : Ma Zihan (FA8Y9M),Liu Hao (YRZNEP)

## Dependencies
The scripts require Python >= 3.8 and the following packages
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/scipylib/index.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [pandas](https://pandas.pydata.org/)
* [matplotlib]( https://matplotlib.org/)

## Dataset
* [Dataset](https://osf.io/nrgx6/)

## What we need to do
First, we performed data preprocessing, created a SpectrogramDataset class using the CNN model, and passed the calculated magnitude as both input and target to the neural network. We optimized hyperparameters, adjusted the learning rate, and used PESQ for audio quality evaluation. Additionally, we utilized matplotlib for visualization. During this process, we encountered several issues: 1. the input size problem in the convolutional layers, which was addressed through multiple adjustments; 2.We replaced the linear layer with an ANN model, then inserted the data from the conformer model to test it and obtained the coefficient of determination results and image visualization

## PESQ
PESQ (Perceptual Evaluation of Speech Quality) is an objective metric for assessing speech signal quality. It compares a degraded speech signal to an original reference signal and provides a score correlating with human perception. The PESQ score ranges from -0.5 to 4.5, with higher scores indicating better quality. A score of 4.5 represents the best quality, while scores below 2.0 indicate significant degradation. PESQ is commonly used to evaluate speech enhancement systems and voice communication systems，We used the coefficient of determination (R²) as an evaluation metric. The closer the R² value is to 1, the better the model's fit, indicating a higher degree of training effectiveness.

## Some training results
