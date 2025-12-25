# Bone Fracture Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This project implements a deep learning-based binary classification system for detecting bone fractures in medical images using the FracAtlas dataset. The model leverages transfer learning with DenseNet121 architecture to achieve accurate fracture detection.

## 🎯 Objectives

- Develop an automated system for fracture detection in X-ray images
- Utilize transfer learning with DenseNet121 pre-trained on ImageNet
- Achieve high accuracy in binary classification (Fractured vs Non-Fractured)
- Provide comprehensive model evaluation metrics

## 📊 Dataset

**FracAtlas Dataset**
- **Source**: Medical imaging dataset for bone fracture detection
- **Classes**: 2 (No Fracture, Fracture)
- **Split Ratio**:
  - Training: 70%
  - Validation: 10%
  - Testing: 20%
- **Total Images**: 817 test images (674 No Fracture, 143 Fracture)

## 🏗️ Model Architecture

### Base Model
- **Architecture**: DenseNet121
- **Pre-trained Weights**: ImageNet
- **Input Shape**: 224×224×3

### Data Augmentation
The model employs comprehensive data augmentation techniques:
- Random Rotation (up to 10°)
- Width/Height Shift (0.1)
- Zoom Range (0.2)
- Shear Transformation (0.1)
- Horizontal Flip
- DenseNet121-specific preprocessing

### Training Configuration
- **Epochs**: 25
- **Batch Size**: 32
- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss Function**: Binary Crossentropy
- **Callbacks**:
  - ModelCheckpoint: Saves best model based on validation loss
  - EarlyStopping: Prevents overfitting

## 📈 Results

### Model Performance

**Test Set Accuracy**: 84%

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No Fracture (0) | 0.85 | 0.99 | 0.91 | 674 |
| Fracture (1) | 0.71 | 0.15 | 0.25 | 143 |
| **Accuracy** | | | **0.84** | **817** |
| **Macro Avg** | 0.78 | 0.57 | 0.58 | 817 |
| **Weighted Avg** | 0.82 | 0.84 | 0.80 | 817 |

### Confusion Matrix

<img width="394" height="354" alt="image" src="https://github.com/user-attachments/assets/ae62d32f-c67d-4997-8f5b-1be5386fb6ee" />


The confusion matrix shows:
- **True Negatives**: 665 (correctly identified non-fractured cases)
- **False Positives**: 9 (non-fractured cases incorrectly classified as fractured)
- **False Negatives**: 121 (fractured cases incorrectly classified as non-fractured)
- **True Positives**: 22 (correctly identified fractured cases)

### Training History

![Training History](training_history.png)

The training curves show model convergence over 25 epochs with both training and validation metrics.

## 🔧 Requirements

```
tensorflow>=2.x
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fracture-detection.git
cd fracture-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the FracAtlas dataset and place it in the `FracAtlas/images` directory

## 💻 Usage

### Training the Model

```python
# Run the Jupyter notebook
jupyter notebook f223111.ipynb
```

The notebook includes:
1. Data loading and preprocessing
2. Model architecture setup
3. Training with data augmentation
4. Model evaluation
5. Visualization of results

### Key Code Sections

**Model Building**:
```python
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
```

**Data Augmentation**:
```python
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True
)
```

## 📊 Key Findings

1. **High Specificity**: The model achieves 99% recall on non-fractured cases, demonstrating excellent ability to identify healthy bones
2. **Low Sensitivity**: The model shows 15% recall on fractured cases, indicating room for improvement in detecting fractures
3. **Class Imbalance**: The dataset has significantly more non-fractured (674) than fractured (143) cases, which may contribute to the lower performance on fracture detection

## 🔮 Future Improvements

1. **Address Class Imbalance**: 
   - Implement oversampling techniques (SMOTE)
   - Use class weights in training
   - Collect more fractured case samples

2. **Model Enhancements**:
   - Experiment with other architectures (ResNet, EfficientNet)
   - Ensemble methods
   - Fine-tune more layers of the base model

3. **Data Augmentation**:
   - Advanced augmentation techniques
   - Test-time augmentation

4. **Evaluation Metrics**:
   - Focus on recall for fractured cases (most critical)
   - Implement focal loss to handle class imbalance



## 👤 Author

**Student ID**: Ahsan Butt

## 🙏 Acknowledgments

- FracAtlas dataset providers
- TensorFlow and Keras teams
- DenseNet121 architecture developers


