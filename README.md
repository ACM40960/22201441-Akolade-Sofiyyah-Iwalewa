[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/foXtNvtG)

## AI IMAGE CLASSIFICATION: ROAD TRANSPORTATION IMAGES
### Project Overview
This project focus is on Convolutional Network Networks (CNNs) for the multiclass classification encompassing six distinct vehicle categories: Ambulance, Bicycle, Bus, Car, Motorcycle and Taxi. The core aim of the poject is the creation of sophistiscated algrothms capable of autonomously discerning and categorizing images into these predefined classes, like human do. The primary goal is to develop an accurate and efficient image classification system that can distinguish between these diverse classes of vehicles.

### Dataset
The dataset employed here is an extract from the 'TAU Vehicle Type Recognition Competition' dataset. Originally comprising 8 vehicle classes within its training folder, I meticulously narrowed it down to 6 distinct classes, thereby assembling a dataset housing a collective sum of 3394 images. This subset was derived from Kaggle's primary training repository, from which I further curated 1127 images and designated them to a dedicated validation folder. Meanwhile, the 'test2' directory encompasses a variable count of images, spanning between 15 to 50 samples, which serves as the canvas for predictive tasks. As an auxiliary component, the 'test' folder features a compilation of 20 randomly selected images, forming a loop through which predictions are generated and subsequently showcased alongside their anticipated classifications.

link to the dataset can be found here: 

### Tools and Technologies Used:

- Programming Language: Python
- Deep Learning Framework: TensorFlow
- Image Preprocessing: Keras ImageDataGenerator
- Hardware Acceleration: NVIDIA GPUs
- Data Visualization: Matplotlib
- Other Libraries: NumPy, Pandas

### Usage and Utilization
To utilize this AI image classification project for road transportation images, follow these steps to set up your environment and run the code: Before you begin, make sure you have Python and pip (Python package installer) installed on your system.
- Open your terminal or command prompt and execute the following command to install the required libraries:pip install numpy matplotlib Pillow tensorflow scikit-learn
This will install the necessary packages: NumPy, Matplotlib, Pillow (PIL), TensorFlow, and scikit-learn.

- For image classification, run this code in your python: python image_classification.py
- 
### Model Architecture
Two architecture of CNN's were considered while creating a model that can accurately classify six distinct classes of vehicles.
#### Model 1:
- The model comprises two convolutional layers followed by a dense layer. 
- The architecture consists of 64 filters in the first convolutional layer and 32 filters in the second convolutional layer. 
- Rectified Linear Unit (ReLU) activation is used to  introduce non-linearity in the network. 
- The output from the second convolutional layer is then flattened and connected to a dense layer with 128 units and ReLU activation.
- Finally, the model ends with an output layer with 6 units (corresponding to the six classes) and a  softmax activation function for multiclass classification.
- To explore the impact of data augmentation on the classification performance, I trained the model using an augmented dataset.Data augmentation is a technique that generates new training samples by applying random transformations to the original images, such as rotation, flipping, and scaling. 
- This augmentation process tends to enhance the model's ability to generalize and improve its performance on unseen data.

![GitHub Logo](https://github.com/ACM40960/22201441-Akolade-Sofiyyah-Iwalewa/blob/main/5.png)
    
#### Model 2:  
- Three successive convolutional layers are implemented.
- The initial layer consists of 64 filters with a receptive field of 3x3, employing the rectified linear unit (ReLU) activation function.
- The subsequent two layers are equipped with 128 and 256 filters respectively, further enhancing feature extraction.
- Following each convolutional layer, a MaxPooling2D layer with a 2x2 pool size is applied to downsample the spatial dimensions, aiding in feature retention and computational efficiency.
- The convolutional outputs are then flattened into a one-dimensional vector, facilitating the transition to fully connected layers.
- A dense layer comprising 128 neurons, activated by ReLU, is introduced to facilitate high-level feature comprehension and integration.
- The final dense layer, comprised of 6 neurons, utilizes the softmax activation function for multiclass classification.
- The model is trained over 100 epochs, with a batch size of 32, on the augmented training dataset.

![GitHub Logo]()
### Model Evaluation

The model's performance was rigorously assessed through a comprehensive set of evaluation metrics, providing a holistic understanding of its capabilities and limitations.

- Learning Curves: Learning curves were employed to visualize the convergence and progression of the training and validation processes over epochs. These curves offer insights into potential overfitting or underfitting scenarios.

- Loss and Accuracy: The training and validation loss functions were closely monitored, offering a dynamic representation of the model's optimization process. Concurrently, training and validation accuracy curves highlighted the model's ability to discriminate between different classes.

- Test Accuracy: The test accuracy metric gauges the model's predictive prowess on unseen data. This statistic offers a reliable measure of the model's generalization ability.

- Precision: Precision, a quintessential metric in multiclass classification, was employed to determine the accuracy of positive predictions. It illuminates the model's effectiveness in correctly identifying true positives among predicted positives.

### Results
#### Model 1


- Confusion Matrix: The confusion matrix provided an intuitive depiction of the model's classification performance across all classes. This matrix unveils the distribution of correct and incorrect predictions, allowing for in-depth analysis of potential areas of confusion.



