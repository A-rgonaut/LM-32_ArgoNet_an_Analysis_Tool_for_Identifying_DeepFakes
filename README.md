# ArgoNet: an Analysis Tool for Identifying DeepFakes

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

> Implementation of a Deep Learning model capable of recognizing DeepFake images, using Python library technologies: opencv, dlib, tensorflow.

## 📖 **Context**

This project was developed for the **Visione Artificiale** examination of Prof. **Liliana Lo Presti**, during the **2024/2025** Academic Year at the **Università degli Studi di Palermo**, **Computer Engineering (LM-32, 2035)** course.

## 👥 **Authors**
_Andrea Spinelli - Antonio Spedito_

## 🛠️ **Technologies Used**

*   **Languages:** Python
*   **Frameworks/Libraries:** opencv, dlib, tensorflow
*   **Other:** Git

## 🚀 **Installation and Startup**

Follow these steps to set up the environment and run the project's modules.

### Prerequisites
*   Python 3.8 or higher
*   Pip (Python package installer)
*   Git

### Installation

First, clone the repository to your local machine:
```bash
git clone https://github.com/A-rgonaut/LM-32_ArgoNet_an_Analysis_Tool_for_Identifying_DeepFakes.git
cd LM-32_ArgoNet_an_Analysis_Tool_for_Identifying_DeepFakes
```

Next, install all the necessary dependencies from the `requirements.txt` file. It is recommended to create a virtual environment to avoid conflicts with other libraries.
```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r [path/to/requirements.txt]
```

### Dataset Structure (for Modules 2 and 3)

The training modules for DeepFake detection require a specific folder structure for the dataset. Ensure that your images are organized as follows:

```
dataset/
|-- real/
|   |-- identity_1/
|   |   |-- image_001.jpg
|   |   |-- ...
|   |-- ...
|
|-- fake/
    |-- identity_1/
    |   |-- image_002.png
    |   |-- ...
    |-- ...
```
**Note on Preprocessing:** Preprocessing is a crucial step. The provided scripts will handle the application of necessary transformations (like LBP for Module 2) before training.

As a starting point, you can use the reference dataset available on Kaggle: **Flickr-Face-HQ and GenAI Dataset (FF-GenAI)**.

### Running the Modules

#### 1. Face Morphing Module
To run the face morphing process, launch the corresponding script. You may need to specify the input images as arguments.
```bash
python [path/to/face_morphing_script.py] --input1 <path/to/image1.jpg> --input2 <path/to/image2.jpg>
```

#### 2. ML Models Module 
To preprocess the dataset with LBP and train the Machine Learning models, run the training script. Make sure to specify the path to the main dataset folder.
```bash
python [path/to/ml_training_script.py] --dataset <path/to/your_dataset_folder>
```

#### 3. ArgoNet Module
To train the custom neural network, ArgoNet, use its dedicated training script, also providing the path to the dataset.
```bash
python [path/to/argonet_training_script.py] --dataset <path/to/your_dataset_folder>
```

## ✨ **Key Features**

This project is divided into three main modules, each with distinct functionalities for analyzing, creating, and detecting DeepFake images.

### 1. Face Morphing Module
*   **Morphed Face Creation:** Generates "morphed" face images by combining the facial features of two source faces.
*   **Process Control:** Provides tools to manage and visualize the morphing process.

### 2. DeepFake Detection with Classic Machine Learning
*   **Feature Extraction with LBP:** Uses the Local Binary Pattern (LBP) algorithm to preprocess images and extract significant texture features, which are effective in distinguishing real images from synthetic ones.
*   **Multiple Model Training:** Trains and evaluates three different Machine Learning algorithms (e.g., Support Vector Machines, Random Forest, etc.) for DeepFake classification.
*   **Comparative Evaluation:** Allows for the comparison of the performance of the different models to identify the most effective approach.

### 3. DeepFake Detection with Custom Neural Network (ArgoNet)
*   **Custom Architecture:** Implements **ArgoNet**, a convolutional neural network (CNN) custom-designed with Multi-Head Self-Attention for the task of DeepFake detection.
*   **End-to-End Training:** Manages the entire pipeline, from image preprocessing to the full training of the network.
*   **High Accuracy:** Aims to achieve high performance and better generalization capabilities compared to classic models.
