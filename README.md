# Machine Learning Classification Algorithms

A beginner-friendly machine learning repository containing implementations of various **classification and clustering algorithms**, primarily using the classic **Iris Dataset**. This repository demonstrates the working principles, mathematical intuition, and performance results of some of the most commonly used supervised and unsupervised learning algorithms.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/status-learning%20log-green)

---

## Table of Contents

- [Dataset](#-dataset)
- [Algorithms](#-algorithms)
  - [1. K-Nearest Neighbors (KNN)](#1-k-nearest-neighbors-knn-classifier)
  - [2. Naive Bayes Classifier](#2-naive-bayes-classifier)
  - [3. Logistic Regression](#3-logistic-regression)
  - [4. Support Vector Machine (SVM)](#4-support-vector-machine-svm)
  - [5. K-Means Clustering](#5-k-means-clustering)
  - [6. Principal Component Analysis (PCA)](#6-principal-component-analysis-pca)
- [Tech Stack](#-tech-stack)

---

## Dataset

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

**Target Classes:** Setosa · Versicolor · Virginica

**Sample Snapshot:**

<p align="center">
  <img width="400" height="407" alt="Iris dataset snapshot" src="https://github.com/user-attachments/assets/9b7ce015-f088-4d4d-a412-63955263e42e" />
</p>

---

## Algorithms

### 1. K-Nearest Neighbors (KNN) Classifier

Classifies a new data point based on the **Euclidean distance** to its neighboring points, assigning the label most common among its nearest neighbors.

**Euclidean Distance Formula:**

<p align="center">
  <img width="394" height="222" alt="Euclidean distance formula" src="https://github.com/user-attachments/assets/59bf2dec-59d3-48c0-81a7-843b96bd699f" />
</p>

**Output:**

<p align="center">
  <img width="739" height="276" alt="KNN output" src="https://github.com/user-attachments/assets/0c1fa6f9-e3e8-4fd8-a69e-3ea00b3aec5c" />
</p>

**Module used:** `sklearn.neighbors`

---

### 2. Naive Bayes Classifier

Uses the extended **Bayes' Rule** to predict the probability of a posterior class **Cₖ** given a feature vector **X**.

**Bayes' Rule:**

<p align="center">
  <img width="347" height="148" alt="Bayes rule formula" src="https://github.com/user-attachments/assets/556b0e3d-ca39-4c9e-9e7b-06c8d06d3538" />
</p>

**Output:**

<p align="center">
  <img width="726" height="264" alt="Naive Bayes output" src="https://github.com/user-attachments/assets/1002b339-2d80-4fb4-b5e0-0a0b9cdf69df" />
</p>

**Module used:** `sklearn.naive_bayes`

---

### 3. Logistic Regression

Fits the input data into a **sigmoid function** to model the probability of class membership.

<p align="center">
  <img width="450" height="337" alt="Logistic regression concept 1" src="https://github.com/user-attachments/assets/43c9e387-c19a-4c7d-8801-83ce904560e0" />
  <img width="450" height="257" alt="Logistic regression concept 2" src="https://github.com/user-attachments/assets/edbeba7b-4f12-41cc-8f42-8b37278844b6" />
</p>

**Output:**

<p align="center">
  <img width="720" height="270" alt="Logistic regression output" src="https://github.com/user-attachments/assets/0c2c810a-9f44-4837-97ab-ede83b45a4c4" />
</p>

**Module used:** `sklearn.linear_model`

---

### 4. Support Vector Machine (SVM)

Finds a **hyperplane** that best separates two (or more) data groups and assigns a label to a new data point based on which side of the hyperplane it falls on.

<p align="center">
  <img width="547" height="365" alt="SVM hyperplane concept" src="https://github.com/user-attachments/assets/747711c4-5a06-4659-b024-9522dc591ae2" />
</p>

**Output:**

<p align="center">
  <img width="701" height="271" alt="SVM output" src="https://github.com/user-attachments/assets/5512ce20-068e-4ad0-966c-47df9e863d42" />
</p>

**Module used:** `sklearn.svm.SVC`

---

### 5. K-Means Clustering

Computes **K clusters** from data, where *K* is pre-defined.

**Steps:**
1. Choose `K` random points to act as centroids.
2. Calculate the distance from each point to every centroid and assign points to the nearest one *(Expectation step)*.
3. Recompute the centroids based on the new cluster assignments *(Maximization step)*.
4. Repeat steps 2–3 until the clusters stabilize.

**Original dataset snapshot:**

<p align="center">
  <img width="849" height="160" alt="K-Means dataset snapshot" src="https://github.com/user-attachments/assets/261b694e-7000-48d8-af9b-2c94ea6b6585" />
</p>

Here, **compactness** and **asymmetry** are used as the parameters for clustering. Compare the data before and after clustering:

| Original Data | Clustered Data |
|---|---|
| <img width="420" alt="Original scatter plot" src="https://github.com/user-attachments/assets/092fce1d-2dcc-4981-8f83-290cbd2eea98" /> | <img width="420" alt="Clustered scatter plot" src="https://github.com/user-attachments/assets/2822870a-cbe7-467e-9cae-7e91104264fb" /> |

**Module used:** `sklearn.cluster.KMeans`

> **Note:** On multi-dimensional data, K-Means is mainly useful for identifying the number of underlying components — it isn't very effective as a standalone clustering method.

**Multi-dimensional output:**

<p align="center">
  <img width="600" alt="Multi-dimensional K-Means output" src="https://github.com/user-attachments/assets/746f1888-4684-4d92-bad1-6fb6818601c9" />
</p>

To analyze multi-dimensional data more efficiently, **Principal Component Analysis (PCA)** can be used to reduce dimensionality before clustering.

---

### 6. Principal Component Analysis (PCA)

PCA reduces multi-dimensional data by projecting it onto a lower-dimensional line/plane. Its main goals are to:

- **Minimize** projection residuals
- **Maximize** variance between the points

<p align="center">
  <img width="500" alt="PCA concept" src="https://github.com/user-attachments/assets/d95cb299-cbdf-4882-a9c4-569bde4e4b47" />
</p>

Using the **Seeds Dataset** from the UCI Machine Learning Repository — originally `(210, 7)`, i.e. 7 dimensions — PCA reduces it down to `(210, 2)`, i.e. 2 dimensions.

**Comparison: K-Means vs. PCA**

**K-Means dataframe after dimensionality reduction with PCA:**

<p align="center">
  <img width="600" alt="K-Means after PCA" src="https://github.com/user-attachments/assets/6d0881ce-117d-46d1-80a2-7d06aaafdbd0" />
</p>

*The image above shows multi-dimensional data reduced to 2D and then clustered using K-Means.*

**Multi-dimensional data via PCA (no clustering applied):**

<p align="center">
  <img width="600" alt="PCA without clustering" src="https://github.com/user-attachments/assets/2ea878fc-03b0-41c3-841e-8be08bcfc695" />
</p>

**Module used:** `sklearn.decomposition.PCA`

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core programming language |
| **scikit-learn** (`sklearn`) | Model implementations & evaluation |
| **Iris Dataset** | Classification benchmark dataset |
| **Seeds Dataset (UCI)** | Clustering / PCA benchmark dataset |
