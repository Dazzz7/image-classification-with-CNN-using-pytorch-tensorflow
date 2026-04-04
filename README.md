# Image Classification using Deep Learning (TensorFlow & PyTorch)
This project focuses on building a robust image classification system to distinguish between cats and dogs using multiple deep learning approaches across both TensorFlow and PyTorch frameworks. The objective is to compare traditional Convolutional Neural Networks (CNNs) with advanced transfer learning models such as InceptionV3, while evaluating their performance, scalability, and generalization capabilities on a real-world dataset.

The dataset used in this project is the Microsoft Cats vs Dogs dataset, consisting of thousands of labeled images of cats and dogs. The data preprocessing pipeline includes loading image paths, labeling, shuffling, and removing corrupted or invalid images to ensure data quality. Exploratory Data Analysis (EDA) is performed by visualizing sample images and class distributions, helping understand dataset balance and variability.

## Data Preprocessing
1. Image resizing to:
(128 × 128) for CNN models
(299 × 299) for Inception models
2. Normalization (rescale = 1/255)
3. Data augmentation:
- Rotation
- Zoom
- Shear
- Horizontal flip
4. Train-test split: 80% training, 20% validation

## Models Implemented
The project is divided into four major models. The first model is a CNN implemented using TensorFlow/Keras, consisting of multiple convolutional and pooling layers followed by dense layers for binary classification. Data augmentation techniques such as rotation, zooming, and flipping are applied to improve model generalization and reduce overfitting. The second model replicates a CNN architecture using PyTorch, with a custom dataset loader designed to handle corrupted images dynamically. This model demonstrates flexibility in handling real-world noisy datasets.

The third model leverages transfer learning using InceptionV3 in TensorFlow. A pretrained model trained on ImageNet is used as a feature extractor, with its layers frozen to retain learned representations. Additional dense layers are added on top to adapt the model to the binary classification task. This approach significantly improves performance by utilizing deep feature hierarchies learned from large-scale datasets. The fourth model implements transfer learning using InceptionV3 in PyTorch, modifying the final fully connected layer for binary classification. The model is trained with GPU acceleration where available and includes handling of auxiliary outputs specific to Inception architectures.

### 🔹 Model 1: CNN (TensorFlow/Keras)
- Convolution + MaxPooling layers
- Fully connected dense layers
- Sigmoid activation for binary classification
- Optimizer: Adam
- Loss: Binary Crossentropy
  
### 🔹 Model 2: CNN (PyTorch)
- Custom CNN architecture with multiple conv layers
- Implemented custom dataset class
- Handled corrupted images dynamically
- Loss: CrossEntropyLoss
- Optimizer: Adam
  
### 🔹 Model 3: Transfer Learning (InceptionV3 - TensorFlow)
- Pretrained InceptionV3 (ImageNet weights)
- Frozen base layers (feature extraction)
- Added custom dense layers on top
- Improved feature representation and accuracy
  
### 🔹 Model 4: Transfer Learning (InceptionV3 - PyTorch)
- Used pretrained InceptionV3 from torchvision
- Modified final fully connected layer for binary classification
- Handled auxiliary outputs
- GPU-supported training

## Training & Evaluation
1. Metrics used:
- Accuracy
- Loss
2. Visualizations:
- Training vs Validation Accuracy
- Training vs Validation Loss
3. Used batch processing and data loaders

## Key Insights
- Transfer learning significantly outperforms basic CNN models
- Inception-based models capture deeper feature representations
- PyTorch and TensorFlow both perform well with proper tuning
- Data augmentation improves generalization and reduces overfitting
- Handling corrupted images is critical in real-world datasets

## Technologies Used
Python
TensorFlow / Keras
PyTorch
OpenCV / PIL
NumPy, Pandas
Matplotlib, Seaborn

Overall, this project demonstrates end-to-end deep learning workflow, including data preprocessing, augmentation, model building, training, evaluation, and comparison across frameworks. It highlights the effectiveness of transfer learning in computer vision tasks and provides practical insights into handling real-world image datasets with noise and inconsistencies.


## Demo-
part 1
https://colab.research.google.com/drive/1oyRLIoxo_TTjhLe7AeeJ0ZBgx8fFndqa#scrollTo=sfGExCvnMoof

part 2
https://colab.research.google.com/drive/1XeznQKrTap1dkn9H98sLAsCZ9DnzDPj7

part 3 
https://colab.research.google.com/drive/1VxyXbJFbLNC6lzWi3XC19B_MWZfiq1iw
