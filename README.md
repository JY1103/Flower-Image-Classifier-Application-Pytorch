# Flower-Image-Classifier-Application-Pytorch

### 1. Installation
The program is running on python 3.6.5, pytorch 0.4.0

### 2. Project Motivation
This code is for Deep Learning course of Data Scientist Nanodegree on Udacity. In this project, a command line application which uses Pytorch, Neural Networks and Transfer Learning is created to train a new neural network classifier on top of a pre-existing model trained on ImageNet to identify different types of flowers.

### 3. File Description
**DataPreprocess.py:** includes all functions to clean the data and process image files. </br>
**train.py:** include functions to build pre-existing model and train model</br>
**predict.py:** include functions to predict flower name with trained model which is saved at checkpoint. </br>
### 4. How to Use
* Run train.py to train a new network
  * Basic usage: python train.py data_directory
  It will print out training loss, validation loss, and validation accuracy as the network trains
  * More options:
    * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    * Choose architecture: python train.py data_dir --arch "vgg13"
    * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    * Use GPU for training: python train.py data_dir --gpu
* Run predict to predict flower name from an image along with the probability of that name
  * Basic usage: python predict.py /path/to/image checkpoint
  * More options:
    * Return top K most likely classes: python predict.py input checkpoint --top_k 3
    * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    * Use GPU for inference: python predict.py input checkpoint --gpu
