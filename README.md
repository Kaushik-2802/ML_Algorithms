## Machine Learning Classification Algorithms
 
A beginner-friendly machine learning repository containing implementations of various **classification algorithms** using the classic **Iris Dataset**. This repository demonstrates the working principles, mathematical intuition, and performance results of some of the most commonly used supervised learning algorithms.
 
---
 
## 📑 Table of Contents
 
- [Dataset](#-dataset)
- [Algorithms](#-algorithms)
  - [K-Nearest Neighbors (KNN)](#1-k-nearest-neighbors-knn-classifier)
  - [Naive Bayes Classifier](#2-naive-bayes-classifier)
  - [Logistic Regression](#3-logistic-regression)
  - [Support Vector Machine (SVM)](#4-support-vector-machine-svm)
---
 
##  Dataset
 
The **Iris Dataset** is one of the most widely used datasets for machine learning classification tasks. It contains **150 samples** belonging to three flower species:
 
- Iris Setosa
- Iris Versicolor
- Iris Virginica
  
**Features:**
 
| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal (cm) |
| Sepal Width | Width of the sepal (cm) |
| Petal Length | Length of the petal (cm) |
| Petal Width | Width of the petal (cm) |
 
**Target Classes:**
 
- Setosa
- Versicolor
- Virginica
**Sample Snapshot:**
 
<img width="400" height="407" alt="image" src="https://github.com/user-attachments/assets/9b7ce015-f088-4d4d-a412-63955263e42e" />
---
 
##  Algorithms
 
### 1. K-Nearest Neighbors (KNN) Classifier
 
Classifies a new data point based on the **Euclidean distance** to its neighboring points, assigning the label most common among its nearest neighbors.
 
**Euclidean Distance Formula:**
 
<img width="394" height="222" alt="image" src="https://github.com/user-attachments/assets/59bf2dec-59d3-48c0-81a7-843b96bd699f" />

**Output:**
 
<img width="739" height="276" alt="image" src="https://github.com/user-attachments/assets/0c1fa6f9-e3e8-4fd8-a69e-3ea00b3aec5c" />

 **Module used:** `sklearn.neighbors`
 
---
 
### 2. Naive Bayes Classifier
 
Uses the extended **Bayes' Rule** to predict the probability of a posterior class **Cₖ** given a feature vector **X**.
 
**Bayes' Rule:**
 
<img width="347" height="148" alt="image" src="https://github.com/user-attachments/assets/556b0e3d-ca39-4c9e-9e7b-06c8d06d3538" />

**Output:**
 
<img width="726" height="264" alt="image" src="https://github.com/user-attachments/assets/1002b339-2d80-4fb4-b5e0-0a0b9cdf69df" />

**Module used:** `sklearn.naive_bayes`
 
---
 
### 3. Logistic Regression
 
Fits the input data into a **sigmoid function** to model the probability of class membership.
 
<img width="4032" height="3024" alt="IMG_4061" src="https://github.com/user-attachments/assets/43c9e387-c19a-4c7d-8801-83ce904560e0" />
<img width="4032" height="2306" alt="IMG_4062" src="https://github.com/user-attachments/assets/edbeba7b-4f12-41cc-8f42-8b37278844b6" />

**Output:**
 
<img width="720" height="270" alt="image" src="https://github.com/user-attachments/assets/0c2c810a-9f44-4837-97ab-ede83b45a4c4" />

 **Module used:** `sklearn.linear_model`
 
---
 
### 4. Support Vector Machine (SVM)
 
Finds a **hyperplane** that best separates two (or more) data groups and assigns a label to a new data point based on which side of the hyperplane it falls on.
 
<img width="547" height="365" alt="image" src="https://github.com/user-attachments/assets/747711c4-5a06-4659-b024-9522dc591ae2" />

**Output:**
 
<img width="701" height="271" alt="image" src="https://github.com/user-attachments/assets/5512ce20-068e-4ad0-966c-47df9e863d42" />

 **Module used:** `sklearn.svm.SVC`
 
---
 
##  Tech Stack
 
- Python
- scikit-learn (`sklearn`)
- Iris Dataset
---
