# CNN Image Classification

> **A deep learning pipeline for accurate image classification using Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset.**

## 📖 Overview

**CNN Image Classification** is a computer vision project built with **PyTorch** that implements a custom Convolutional Neural Network to categorize images into 10 distinct classes.

The project encompasses the full machine learning lifecycle—from data acquisition and augmentation to model architecture design, training optimization, and final performance evaluation. It serves as a robust implementation of standard deep learning practices for multi-class classification tasks.

---

## ✨ Key Features

### 🧠 Deep Learning Architecture

* **Custom CNN Design:** Features a modular architecture with three convolutional blocks, utilizing **3x3 kernels** and **Max Pooling** for effective feature extraction and dimensionality reduction.
* **Regularization:** Implements **Dropout** layers to mitigate overfitting and **Batch Normalization** to stabilize and accelerate training convergence.
* **Adaptive Layers:** Uses `AdaptiveAvgPool2d` to handle variable input sizes effectively before the fully connected classification head.

### 🔄 Data Pipeline

* **Automated Preprocessing:** Automatically handles the downloading, extraction, and normalization of the **CIFAR-10** dataset within the notebook.
* **Data Augmentation:** Applies real-time transformations like **Random Horizontal Flip** and **Random Rotation** to enhance model generalization.
* **Efficient Loading:** Leverages PyTorch `DataLoader` with `SubsetRandomSampler` for seamless training and validation splitting.

### 📊 Training & Analysis

* **Optimization:** Trained using **Stochastic Gradient Descent (SGD)** with momentum and weight decay.
* **Checkpointing:** Automatically tracks validation loss and generates the best-performing model state (`best_cifar10_cnn.pt`) during training.
* **Detailed Metrics:** Provides overall test accuracy (~84%) along with granular class-wise performance breakdown.
* **Visual Inference:** Visualizes prediction results, identifying both correctly classified and misclassified examples with confidence scores.

---

## 📂 Project Structure

```text
Image_Classification_CNN/
├── Image_Classification_CNN.ipynb  # Main Jupyter Notebook source code
└── README.md                       # Project Documentation

```

> **Note:** The dataset directory (`data/`) and the model checkpoint (`best_cifar10_cnn.pt`) will be automatically generated locally when you run the notebook.

---

## 🚀 Getting Started

### Prerequisites

* **Python 3.8+**
* **Jupyter Notebook** / JupyterLab
* **PyTorch** (with `torchvision`)
* **NumPy** & **Matplotlib**

### Installation

1. Clone the repository:
```bash
git clone https://github.com/andyhu11/Project-Documentation.git

```


2. Navigate to the project directory:
```bash
cd Project-Documentation/Image_Classification_CNN

```


3. Install dependencies (if using pip):
```bash
pip install torch torchvision numpy matplotlib

```



### Usage Guide

1. **Launch:** Open the notebook to view the workflow.
```bash
jupyter notebook Image_Classification_CNN.ipynb

```


2. **Run:** Execute the cells sequentially. The notebook is self-contained and will:
* **Download** the CIFAR-10 dataset to a local `data/` folder.
* **Initialize** the `Net` architecture.
* **Train** the model for 30 epochs (and save `best_cifar10_cnn.pt` locally).
* **Visualize** accuracy statistics and predictions.



---

## 📈 Performance

The model demonstrates strong performance on the test set:

* **Overall Accuracy:** **84.63%**
* **Top Classes:**
* *Automobile:* 95.80%
* *Ship:* 89.70%


* **Loss:** Achieved a best validation loss of **0.4442**.

---

## 🤝 Contributing

Contributions to improve accuracy or optimize architecture are welcome:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewArchitecture`).
3. Commit your Changes (`git commit -m 'Add ResNet block'`).
4. Push to the Branch (`git push origin feature/NewArchitecture`).
5. Open a Pull Request.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
