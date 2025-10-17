# Brain Tumor Classification using InceptionV3

This project implements a **brain tumor detection system** using deep learning on MRI images. It classifies brain MRI images into **two classes**: `tumor` (yes) and `no tumor`. The model leverages **InceptionV3** with transfer learning and applies preprocessing techniques like **Gaussian denoising** and **data augmentation** for robust performance.

## Dataset

The dataset consists of MRI images stored in two folders:

- `no` – images without tumors  
- `yes` – images with tumors  

All images are resized to **128x128 pixels**.  

> Images are loaded, normalized, and labeled for binary classification.

## Preprocessing

- Images are converted to float32 and scaled to [0,1].  
- Gaussian blur is applied for **denoising**.  
- Data augmentation includes rotation, width/height shift, shear, zoom, and horizontal flip.

## Model Architecture

- **Base model:** InceptionV3 (pre-trained on ImageNet, layers frozen)  
- **Custom layers:**  
  - GlobalAveragePooling2D  
  - Dropout (0.5)  
  - Dense layer with 2 output units (softmax activation)  

- **Optimizer:** Adam  
- **Loss:** Categorical Crossentropy  
- **Metrics:** Accuracy  

### Model Parameters
- **Total params:** 21,815,080 (83.22 MB)
- **Trainable params:** 4,098 (16.01 KB) 
- **Non-trainable params:** 21,802,784 (83.17 MB) 
- **Optimizer params:** 8,198 (32.03 KB)

## Training

- **Batch size:** 32  
- **Epochs:** 10  
- **Callbacks:** ReduceLROnPlateau to reduce learning rate when validation loss plateaus.  
- **Data augmentation** applied during training.

## Results

- **Test Accuracy:** 94.12%  
- The model shows strong performance in binary brain tumor classification.

