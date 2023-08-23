[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/foXtNvtG)

## AI IMAGE CLASSIFICATION: ROAD TRANSPORTATION IMAGES
### Project Overview
This project focus is on Convolutional Network Networks (CNNs) for the multiclass classification encompassing six distinct vehicle categories: Ambulance, Bicycle, Bus, Car, Motorcycle and Taxi. The core aim of the poject is the creation of sophistiscated algrothms capable of autonomously categorizing images into these predefined classes, like human do. The primary goal is to develop an accurate and efficient image classification system that can distinguish between these diverse classes of vehicles.

### Dataset
The dataset employed here is an extract from the 'TAU Vehicle Type Recognition Competition' dataset. Originally comprising 8 vehicle classes within its training folder, I meticulously narrowed it down to 6 distinct classes, I had the total of 3394 images in my training folder. I also included 1127 images from kaggles training data as my validation images in the validation folder. I also have a 'test2' folder which contains between 15 to 50 samples for each class, which serves as the canvas for predictive tasks. As an auxiliary component, the 'test' folder features a compilation of 20 randomly selected images, forming a loop through which predictions are generated and subsequently showcased alongside their anticipated classifications.

link to the dataset can be found here: https://drive.google.com/file/d/1br2f-h5LIyjCAa8CqX0e6TD-10OJbzbD/view?usp=drive_link

### Usage

- Programming Language: Python 3.7

#### Installation
![GitHub Logo](https://github.com/ACM40960/22201441-Akolade-Sofiyyah-Iwalewa/blob/main/3.png)

  
#### Python libararies 
![GitHub Logo](https://github.com/ACM40960/22201441-Akolade-Sofiyyah-Iwalewa/blob/main/1.png)

### Model Architecture
Two architecture of CNN's were considered while creating a model that can accurately classify six distinct classes of vehicles. Both models was trained using data augumentation.

#### Data Augumentation

![GitHub Logo](https://github.com/ACM40960/22201441-Akolade-Sofiyyah-Iwalewa/blob/main/2.png)

#### First Model:

![GitHub Logo](https://github.com/ACM40960/22201441-Akolade-Sofiyyah-Iwalewa/blob/main/5.png)
    
#### Second Model :  

![GitHub Logo](https://github.com/ACM40960/22201441-Akolade-Sofiyyah-Iwalewa/blob/main/6.png)

### Model Evaluation

The model's performance was rigorously assessed through a comprehensive set of evaluation metrics, providing a deep understanding of its capabilities and limitations.

- Learning Curves: Learning curves were employed to visualize the convergence and progression of the training and validation processes over epochs. These curves offer insights into potential overfitting or underfitting scenarios.

- Loss and Accuracy: The training and validation loss functions were closely monitored, offering a dynamic representation of the model's optimization process. Concurrently, training and validation accuracy curves highlighted the model's ability to discriminate between different classes.

- Test Accuracy: The test accuracy metric gauges the model's predictive prowess on unseen data. This statistic offers a reliable measure of the model's generalization ability.

- Precision: Precision was employed to determine the accuracy of positive predictions. It illuminates the model's effectiveness in correctly identifying true positives among predicted positives.

- Confusion Matrix: The confusion matrix provided an intuitive depiction of the model's classification performance across all classes. This matrix unveils the distribution of correct and incorrect predictions, allowing for in-depth analysis of potential areas of confusion.

- Randomly predicting and displaying images in the "test folder".
  
### Results

#### First Model 

![GitHub Logo](https://github.com/ACM40960/22201441-Akolade-Sofiyyah-Iwalewa/blob/main/A1.png)

![GitHub Logo](https://github.com/ACM40960/22201441-Akolade-Sofiyyah-Iwalewa/blob/main/A2.png)

- The learning curves demonstrate that the model trained with data augmentation performs well on the validation set, indicating minimal overfitting. The stable training process suggests that the Adam optimizer is a suitable choice for this task.
- The final training accuracy is approximately 69.39%. This means the model correctly classified about 69.39% of the training samples. 
- Validation Accuracy: The final validation accuracy is around 71.07%. This indicates that the model achieved an accuracy of . 71.07% approximately on the validation set, which consists of unseen data.
- The final training loss is 0.7746. The training loss is a measure of how well the model's predictions match the ground truth labels during training. A lower training loss indicates that the model's predictions are closer to the actual labels, which is a positive sign.
- The final validation loss is 0.7914. The validation loss measures how well the model generalizes to unseen data. Similar to training loss, a lower validation loss indicates 
better generalization performance.
- The test accuracy of the model is approximately 60.67%, indicating that it correctly classified around 60.67% of the test samples.
- The precision of the model is 0.6411, which indicates that, on average when the model predicts a sample to belong to a certain class, it is correct approximately 64.11% of the time.

![GitHub Logo](https://github.com/ACM40960/22201441-Akolade-Sofiyyah-Iwalewa/blob/main/7.png)

 From the confusion matrix, These are the observations: 
 
- Ambulance: The model correctly predicted 3 samples as ambulance (true positives), while incorrectly predicting them as Classes 2, 3, 4, and 5. It also predicted 4 amnulances samples as bus, 8 as car, and 5 as motorcycle. 

- Bicycle: The model correctly predicted 22 samples as bicycle (true positives), but incorrectly predicted 1 bicycle sample as car, and 27 as motorcycle. This suggests some misclassification between bicycle and other classes, particularly motorcycle.

- Bus: The model correctly predicted 41 samples as bus (true positives), but incorrectly predicted 2 bus samples as ambulance, 4 as car, and 2 as motorcycle.

- Car: The model correctly predicted 25 samples as car (true positives), but incorrectly predicted 1 car sample as bicycle, 4 as bus, 14 as motorcycle, and 6 as taxi. 

- Motorcycle: The model correctly predicted 43 samples as motorcycle (true positives), but incorrectly predicted 3 motorcycle samples as bicycle, 1 as bus, 2 as car, and 1 as taxi.
  
- Taxi: The model correctly predicted 11 samples as taxi (true positives), but incorrectly predicted 2 taxi samples as ambulance, 4 as bus, and 2 as car. 

![GitHub Logo](https://github.com/ACM40960/22201441-Akolade-Sofiyyah-Iwalewa/blob/main/A3.png)

- The model was used in predicting 20 random images, it appears 3 images were misclassified.

#### Second Model 

![GitHub Logo](https://github.com/ACM40960/22201441-Akolade-Sofiyyah-Iwalewa/blob/main/A4.png)

![GitHub Logo](https://github.com/ACM40960/22201441-Akolade-Sofiyyah-Iwalewa/blob/main/A5.png)

- The learning curves demonstrate that the model trained with three CNN's exhbit some overfitting.
- The final training accuracy is approximately 90.4%. This means the model correctly classified about 90.4% of the training samples. 
- Validation Accuracy: The final validation accuracy is approximately 73.38%. This indicates that the model achieved an accuracy of . 73.38% approximately on the validation set, which consists of unseen data.
- The final training loss is 0.0.2602. The model has a lower training loss indicates that the model's predictions are closer to the actual labels, which is a positive sign.
- The final validation loss is 1.03006. The validation loss measures how well the model generalizes to unseen data. The validation loss is not high. 
- The test accuracy of the model is approximately 66%, indicating that it correctly classified around 6% of the6 test samples.
- The precision of the model is 0.6769, which indicates that, on average when the model predicts a sample to belong to a certain class, it is correct approximately 67.7% of the time.


![GitHub Logo](https://github.com/ACM40960/22201441-Akolade-Sofiyyah-Iwalewa/blob/main/8.png)

 From the confusion matrix: These are the observations 

- Ambulance: The model correctly predicted 6 samples as ambulance, while misclassifying 4 as bus, 9 as car, and 1 as motocycle. 
- Bicycle: The model correctly predicted 41 samples as bicycle, while misclassifying 2 as car and 7 as motorcycle. This suggests a good performance in identifying bicycle, with relatively minor confusion.
- Bus: The model correctly predicted 30 samples as bus, but misclassified 2 as Class ambulance, 13 as car, and 4 as taxi. 
- Car: The model correctly predicted 38 samples as car, while misclassifying 1 as ambulance, 3 as bicycle, 3 as bus, and 2 as taxi. 
- Motorcycle: The model correctly predicted 30 samples as motocycle, but misclassified 13 as bicycle, 2 as bus, and 5 as car.
- Taxi: The model correctly predicted 13 samples as Taxi , while misclassifying 1 as ambulance, 5 as car, and 5 as motocycle. 

![GitHub Logo](https://github.com/ACM40960/22201441-Akolade-Sofiyyah-Iwalewa/blob/main/A6.png)

- The model was used in predicting 20 random images, it appears only an image was misclassified..

### References
1) Machine learning and AI, Dr. Micheal Fop,UCD.
2) Image Classification Based On CNN: A Survey Ahmed A. Elngar1, Mohamed Arafa2, Amar Fathy3, Basma Moustafa4 ,Omar Mahmoud5 and Mohamed Shaban6*, Nehal Fawzy
3) S. Albawi, T. A. Mohammed and S. Al-Zawi, ”Understanding of a convolutional neural network,” 2017 International Conference on Engineering andTechnology (ICET), Antalya, Turkey, 2017, pp. 1-6, doi: 10.1109/ICEngTechnol.2017.8308186.

### Future Work
In the nearest future, my aspiration involves training more diverse model architectures while working with a more samples of training and validation data. This endeavor aims to explore alternative neural network structures, that might lead to better accuracy and precision broader applicability. I also would provide a better representation of the confusion matrix.



