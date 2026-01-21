# CNN Image Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

> **A robust deep learning pipeline utilizing a custom Convolutional Neural Network (CNN) to achieve high-accuracy image classification on the CIFAR-10 dataset.**

## 📖 Overview

**CNN Image Classification** is a computer vision project designed to demonstrate the end-to-end lifecycle of a deep learning application. Built with **PyTorch**, this project implements a custom 3-layer CNN architecture to categorize images into 10 distinct classes (e.g., Airplanes, Cars, Birds) with high precision.

Moving beyond simple Multi-Layer Perceptrons (MLPs), this solution leverages spatial feature extraction techniques including **Convolution**, **Max Pooling**, and **Batch Normalization**. The pipeline encompasses rigorous data augmentation, model checkpointing, and granular performance analysis, serving as a reference implementation for standard computer vision tasks.

---

## ✨ Key Features

### 🧠 Deep Learning Architecture
* **Custom CNN Backbone:**
    * Features a modular design with **3 Convolutional Blocks** (Conv2d -> BatchNorm -> ReLU -> MaxPool).
    * Utilizes **3x3 Kernels** with padding to preserve spatial dimensions during feature extraction.
* **Regularization & Stability:**
    * Implements **Dropout (p=0.25)** to randomly zero out neurons, effectively mitigating overfitting.
    * **Batch Normalization** layers are integrated to accelerate convergence and stabilize the learning process.
* **Adaptive Pooling:**
    * Leverages `AdaptiveAvgPool2d` to handle variable input sizes seamlessly before the fully connected classification head.

### 🔄 Data Engineering Pipeline
* **Automated ETL:** Automatically handles the extraction, transformation, and loading (ETL) of the **CIFAR-10** dataset upon execution.
* **Robust Augmentation:**
    * **Random Horizontal Flip** & **Random Rotation** ($10^\circ$) to improve model generalization on unseen data.
    * Standardization using channel-wise Mean and Std deviation.
* **Efficient Loading:** Optimized `DataLoader` with `SubsetRandomSampler` for reproducible Train/Validation splits (20% validation).

### 📊 Training & Optimization
* **Optimizer:** Trained using **Stochastic Gradient Descent (SGD)** with Nesterov Momentum (0.9) for escaping local minima.
* **Checkpointing:** Implements a logic to track Validation Loss and automatically save the best model state (`best_cifar10_cnn.pt`) locally.
* **Learning Rate Scheduler:** Configured with a fixed LR of 0.01, optimized for 30 epochs of training.

---

## 📈 Performance

Based on the evaluation of the **CIFAR-10** test set (10,000 images), the model demonstrates strong generalization capabilities, achieving an overall accuracy of **84.63%**.

### Training Dynamics & Visualization

> **Note:** The model achieved its best validation loss of **0.4442** during the training phase, indicating minimal overfitting. Detailed visualization charts (Loss curves, Misclassified examples) are rendered directly within the Jupyter Notebook output cells.

### Class-wise Evaluation Metrics
| Training Loss Curve | Prediction Samples |
| :---: | :---: |
| <img src="images/training_curve.png" width="100%" alt="Training Loss Curve"> | <img src="images/prediction_sample.png" width="100%" alt="Model Predictions"> |

The table below breaks down the model's performance across individual categories, highlighting its strength in identifying mechanical objects vs. biological subjects.

| Class Category | Accuracy | Performance Tier |
| :--- | :--- | :--- |
| **Automobile** | **95.80%** | 🟢 High Confidence |
| **Frog** | **93.80%** | 🟢 High Confidence |
| **Truck** | **93.70%** | 🟢 High Confidence |
| **Ship** | **89.70%** | 🟢 High Confidence |
| **Airplane** | **89.20%** | 🟢 High Confidence |
| **Horse** | **86.40%** | 🟢 High Confidence |
| *Bird* | *76.80%* | 🟡 Moderate |
| *Cat* | *69.90%* | 🔴 Hardest Class |

> **Insight:** The model performs exceptionally well on rigid objects (Cars, Trucks) but faces slight challenges with deformable objects (Cats, Birds), suggesting potential for improvement via deeper architectures like ResNet.

---

## 📂 Project Structure

```text
Image_Classification_CNN/
├── Image_Classification_CNN.ipynb  # Main Jupyter Notebook source code
└── README.md                       # Project Documentation

```

> **Runtime Artifacts:**
> When you run the notebook, the following will be generated locally:
> * `data/`: Folder containing the downloaded CIFAR-10 dataset.
> * `best_cifar10_cnn.pt`: The saved model weights with the lowest validation loss.
> 
> 

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* PyTorch (with `torchvision`)
* NumPy & Matplotlib
* Jupyter Notebook

### Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/your-username/Image_Classification_CNN.git](https://github.com/your-username/Image_Classification_CNN.git)
cd Image_Classification_CNN

```


2. **Install dependencies:**
```bash
pip install torch torchvision numpy matplotlib jupyter

```



### Usage Guide

1. **Launch Jupyter Notebook:**
```bash
jupyter notebook Image_Classification_CNN.ipynb

```


2. **Execute the Pipeline:**
Run all cells sequentially. The notebook is self-contained and will:
* **Download** the CIFAR-10 dataset.
* **Train** the CNN for 30 epochs (approx. 20-30 mins on GPU).
* **Save** the best model weights.
* **Visualize** accuracy statistics and predictions.



---

## 🚧 Future Roadmap

* **Architecture Upgrade:** Implement **ResNet-18** or **VGG-16** via Transfer Learning to push accuracy above 90%.
* **Hyperparameter Tuning:** Integrate **Optuna** to automatically find the optimal Learning Rate and Batch Size.
* **Deployment:** Export the model to **ONNX** format for cross-platform inference.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improving the accuracy on "Cat" or "Bird" classes:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewArchitecture`).
3. Commit your Changes (`git commit -m 'Add Residual Blocks'`).
4. Push to the Branch.
5. Open a Pull Request.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

```

```
