# AI-Attendance-System-and-Gender-Detection
# Class Attendance using AI

## Project Overview
This AI-based attendance system automates student identification and tracking in classrooms using Machine Learning (ML) and Deep Learning (DL). It detects faces, identifies students, and classifies gender from classroom images.

## Author
**Ubaid Ur Rehman**  
Supervised By: **Prof. Michel Riveill, Diane Lingrand**  
Institution: **University Côte d’Azur**  
Date: **March 30, 2025**  

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Face Detection](#face-detection)
4. [Machine Learning Approach](#machine-learning-approach)
5. [Deep Learning Approach](#deep-learning-approach)
6. [Gender Classification](#gender-classification)
7. [Ethical Considerations](#ethical-considerations)
8. [Installation & Usage](#installation--usage)
9. [Contact](#contact)

## Introduction
Traditional attendance tracking is time-consuming and prone to errors. This project automates attendance marking by capturing classroom images, detecting faces, recognizing students, and updating attendance records.

## Methodology
### Development Tools
- **Programming Language**: Python
- **Libraries**: OpenCV, TensorFlow, Keras, Scikit-Learn
- **Hardware**: Windows 11, RTX 3050 GPU

### Workflow
1. **Data Collection & Preprocessing**: Extracting and augmenting images.
2. **Face Detection**: Haar Cascade classifiers for facial recognition.
3. **Student Identification**: ML (SVM) and DL (CNN) models.
4. **Attendance Marking**: Automated updates based on detection.
5. **Gender Classification**: CNN-based models for gender prediction.
6. **Ethical Considerations**: Addressing privacy, bias, and computational constraints.

## Face Detection
- **Grayscale Conversion**: Improves detection accuracy.
- **Haar Cascade Classifier**: Detects faces and extracts regions of interest.

## Machine Learning Approach
- **Feature Extraction**: Using Wavelet Transform.
- **SVM Model**: Hyperparameter tuning via Halving Grid Search CV.

## Deep Learning Approach
- **Pre-trained Models**: VGG16, InceptionV3, ResNet50, EfficientNetB0.
- **Data Augmentation**: Enhances model robustness.
- **Evaluation**: Accuracy, confusion matrix, loss plots.

## Gender Classification
- **Training on External Dataset**: CNN-based classification.
- **Evaluation Metrics**: Accuracy, confusion matrices.

## Ethical Considerations
- **Bias & Diversity**: Ensuring dataset fairness.
- **Privacy & Security**: Addressing facial recognition concerns.
- **Computational Efficiency**: Optimizing model performance.




## Contact
For queries, contact me at [ubaidfr404786@gmail.com](mailto:ubaidfr404786@gmail.com).

