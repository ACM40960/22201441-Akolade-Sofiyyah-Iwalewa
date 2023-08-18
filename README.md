[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/foXtNvtG)

## AI IMAGE CLASSIFICATION: ROAD TRANSPORTATION IMAGES
### Project Overview
This project focus is on Convolutional Network Networks (CNNs) for the multiclass classification encompassing six distinct vehicle categories: Ambulance, Bicycle, Bus, Car, Motorcycle and Taxi. The core aim of the poject is the creation of sophistiscated algrothms capable of autonomously discerning and categorizing images into these predefined classes,
like human do. The primary goal is to develop an accurate and efficient image classification system that can distinguish between these diverse classes of vehicles.

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
  
#### Model 2:  
