# Image Classification with CNN (CIFAR-10)

> An example project that trains and evaluates a Convolutional Neural Network (CNN) on the **CIFAR-10** dataset using **PyTorch**.  
> This project contains only one file: `Image_Classification_CNN.ipynb`

---

## 1. Project Overview

This project demonstrates a complete image classification pipeline in a Jupyter Notebook, covering **environment check → data download/preprocessing → CNN model construction → training/validation → saving the best model → test evaluation → inference & visualization**.

- Dataset: CIFAR-10 (10 classes, 32×32 color images)
- Framework: PyTorch / torchvision
- Training outputs: training logs, best model weights file `best_cifar10_cnn.pt`
- Evaluation outputs: overall test accuracy and per-class accuracy, plus visualized inference results (correct/incorrect samples)

---

## 2. Features

### 2.1 Environment Check
- Print PyTorch version: `torch.__version__`
- Generate a random tensor `torch.rand(2, 3)` to verify the runtime works properly
- Check whether CUDA is available (automatically select `cuda` or `cpu`)

### 2.2 Data Download and Preprocessing
- Automatically download CIFAR-10 to a local directory using `torchvision.datasets.CIFAR10(..., download=True)`
- Data augmentation and normalization (example):
  - `RandomHorizontalFlip()`
  - `RandomRotation(10)`
  - `ToTensor()`
  - `Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))`
- Create a validation split:
  - `valid_size = 0.2`
  - Use `SubsetRandomSampler` to build `train_loader / valid_loader`
- Build the test loader: `test_loader`
- The 10 class names:
  - `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

### 2.3 CNN Model
- Three convolutional blocks (Conv + BN + ReLU + Conv + BN + ReLU + MaxPool)
- Global pooling via `AdaptiveAvgPool2d((2, 2))`
- Classification head: Dropout + Linear + ReLU + Dropout + Linear (outputs 10 classes)

### 2.4 Training and Saving the Best Model
- Loss function: `CrossEntropyLoss`
- Optimizer: `SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)`
- Train for `n_epochs = 30` (adjustable)
- Save the best model (based on the lowest validation loss) to: `best_cifar10_cnn.pt`

### 2.5 Test Evaluation
- Report test Loss / Accuracy
- Report per-class accuracy for all 10 classes

### 2.6 Inference and Visualization
- Single-image inference: output the ground-truth label, predicted label, and confidence score
- Visualization:
  - Display several correctly predicted examples
  - Display several incorrectly predicted examples

---

## 3. Environment Requirements

- **Python**: 3.10 (the notebook example uses `3.10.19`)
- **Main dependencies**
  - `torch`
  - `torchvision`
  - `numpy`
  - `matplotlib`
- **Hardware**
  - Runs on CPU; if you install a CUDA-enabled PyTorch build, it will automatically use GPU (depending on `torch.cuda.is_available()`)

---

## 4. Quick Start

### 4.1 Install Dependencies (conda example)
```bash
conda create -n cnn-cifar10 python=3.10 -y
conda activate cnn-cifar10
conda install pytorch torchvision torchaudio -c pytorch
conda install numpy matplotlib

### 4.2 Ensure the CIFAR-10 Data is Ready:
- The Notebook uses the default `data_root = ./data`
- The code sets `download=False`, so you need to have the CIFAR-10 data prepared in advance (or change the code to `download=True`)

### 4.3 Open and Run the Notebook:
- Start Jupyter:
  ```bash
  jupyter notebook
  ```
- Open: `Image_Classification_CNN.ipynb`

### 4.4 Execute the Cells in Order:
- Extract data → Build the DataLoader → Define the model → Train and save the best model → Test and evaluate → Visualize the inference results

### 4.5 After Training, the Following Will Be Generated:
- `best_cifar10_cnn.pt` (Model weights corresponding to the best validation loss)

---

## 5. Reference Results (Example of Notebook Output)

- Best validation loss (example): `Best validation loss: 0.4442`
- Test accuracy (example): `Test Accuracy: 84.63%`
- Example per-class accuracy (partial):
  - automobile: 95.80%
  - cat: 64.00%
  - ship: 89.70%
  - truck: 89.60%

> Results may slightly vary depending on the random seed, data augmentation, hardware, and environment.

---

## 6. Frequently Asked Questions (FAQ)

### Q1: Error "Dataset not found / download=False" at Runtime
In the Notebook, `datasets.CIFAR10(... download=False ...)` will not automatically download the data. You can:
- Manually place the CIFAR-10 data into the corresponding `./data` directory; or
- Change `download=False` to `download=True` (the data will be downloaded on first run)

### Q2: Encountering FutureWarning for `torch.load` (weights_only)
When the Notebook loads the model, it uses a command similar to:
```python
torch.load('best_cifar10_cnn.pt', map_location=device)
```
```
