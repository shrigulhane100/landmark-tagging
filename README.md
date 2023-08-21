# Landmark Classification & Tagging for Social Media 📷🗺️

Welcome to the Landmark Classification project! In this exciting endeavor, we'll be delving into the world of computer vision, specifically tackling the challenge of automatically predicting the location of images based on the landmarks depicted in them. Through the application of Convolutional Neural Networks (CNNs), transfer learning, and various image processing techniques, we'll create models capable of recognizing landmarks and enhancing the user experience on photo sharing platforms. 🌍📸

https://github.com/shrigulhane100/landmark-tagging/blob/main/static_images/test/09.Golden_Gate_Bridge/190f3bae17c32c37.jpg?raw=true

## Project Overview 🏞️

In this project, Task will be building a landmark classifier to predict the location of images based on the landmarks present in them. The goal is to provide meaningful tags and location information for photos that lack location metadata, ultimately enhancing the user experience on photo sharing platforms. We'll embark on the machine learning design process, encompassing data preprocessing, CNN design and training, transfer learning, model evaluation, and the development of a user-friendly app. The skills applied include CNN fundamentals, transfer learning, autoencoders, object detection, and object segmentation.

### Motivation behind the project

Photo sharing services often lack location data for uploaded photos, such asautomatic suggestion of relevant tags. Sometimes many photos uploaded to these services will not have location metadata available because smartphone does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.
One way to infer the location is to detect and classify landmarks in the image. This project will build a CNN-powered app to automatically predict the location of an image based on landmarks and suggest the top k most relevant landmarks from 50 possible landmarks from across the world.


#### Setting up locally

This setup requires a bit of familiarity with creating a working deep learning environment. While things should work out of the box, in case of problems you might have to do operations on your system (like installing new NVIDIA drivers) that are not covered in the class. Please do this if you are at least a bit familiar with these subjects, otherwise please consider using the provided Udacity workspace that you find in the classroom.

1. Open a terminal and clone the repository, then navigate to the downloaded folder:
	
	```	
		git clone https://github.com/udacity/cd1821-CNN-project-starter.git
		cd cd1821-CNN-project-starter
	```
    
2. Create a new conda environment with python 3.7.6:

    ```
        conda create --name udacity_cnn_project -y python=3.7.6
        conda activate udacity_cnn_project
    ```
    
    NOTE: you will have to execute `conda activate udacity_cnn_project` for every new terminal session.
    
3. Install the requirements of the project:

    ```
        pip install -r requirements.txt
    ```

4. Install and open Jupyter lab:
	
	```
        pip install jupyterlab
		jupyter lab
	```



## I. Create a CNN to Classify Landmarks from Scratch 🖼️

### 1. Data Loading and Exploration 📊

- Prepared data for neural networks, incorporating image preprocessing and augmentation techniques.
- Utilized PyTorch's DataLoader to efficiently feed training data.
- Applied normalization to enhance model convergence.
- Visualized a batch of images to gain insights into the data.

### 2. Model Design and Training 🧠

- Designed and implemented a CNN architecture for image classification from scratch.
- Chose an appropriate loss function, optimizer, and learning rate scheduler.
- Trained the CNN model using the training dataset.
- Explored different architecture configurations and layer types to enhance model performance.

### 3. Model Testing and Evaluation 🧪

- Evaluated the trained model's performance using the holdout set.
- Fine-tuned hyperparameters and model architecture to achieve higher accuracy.
- Achieved a test accuracy of **56%** by iteratively refining the CNN model over **120 epochs**.

### 4. Export the Model 🔗

- Exported the trained model using TorchScript for deployment.

## II. Use Transfer Learning 🚀

### 1. Create Transfer Learning Architecture 🏗️

- Implemented transfer learning using the **ResNet18** architecture.
- Justified the choice of ResNet18 and leveraged its pre-trained weights.
- Achieved a test accuracy of **72%** by leveraging the power of transfer learning.

### 2. Export the Model 🔗

- Exported the transfer learning model using TorchScript for deployment.

## III. Write Your Application 📱

- Developed a simple application for testing the trained models.
- Uploaded images of landmarks to the app to observe model behavior.
- The app displayed the top 5 predicted classes for each uploaded image, providing insights into the model's performance.

#### 📷🌆
