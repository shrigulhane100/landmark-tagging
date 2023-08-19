# Landmark Classification & Tagging for Social Media 📷🗺️

Welcome to the Landmark Classification project! In this exciting endeavor, we'll be delving into the world of computer vision, specifically tackling the challenge of automatically predicting the location of images based on the landmarks depicted in them. Through the application of Convolutional Neural Networks (CNNs), transfer learning, and various image processing techniques, we'll create models capable of recognizing landmarks and enhancing the user experience on photo sharing platforms. 🌍📸

## Project Overview 🏞️

In this project, we'll be building a landmark classifier to predict the location of images based on the landmarks present in them. The goal is to provide meaningful tags and location information for photos that lack location metadata, ultimately enhancing the user experience on photo sharing platforms. We'll embark on the machine learning design process, encompassing data preprocessing, CNN design and training, transfer learning, model evaluation, and the development of a user-friendly app. The skills applied include CNN fundamentals, transfer learning, autoencoders, object detection, and object segmentation.

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

Feel free to reach out for questions or insights. Happy coding and landmark-classifying! 📷🌆
