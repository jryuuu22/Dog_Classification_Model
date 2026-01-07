# End-to-End Multi-Class Dog Breed Classification (TensorFlow + TF Hub)

This project builds an end-to-end **multi-class image classifier** to predict a dog’s breed from an input image using **TensorFlow 2**, **Keras**, and **TensorFlow Hub**. The model is trained and evaluated using Kaggle’s **Dog Breed Identification** dataset (120 breeds).

## Problem
Given an image of a dog, predict the correct breed (**120-class classification**).

## Dataset
Source: Kaggle Dog Breed Identification competition  
- ~10,222 labeled training images  
- ~10,357 unlabeled test images  
- Labels provided as breed names; test set requires submitting **probability distributions** across all breeds.

## Approach
- **Transfer learning** with **MobileNetV2 (ImageNet-pretrained)** from TensorFlow Hub
- Custom preprocessing pipeline:
  - Read image bytes from filepaths
  - Decode JPEG → RGB tensor
  - Normalize pixel values to `[0, 1]`
  - Resize to **224 × 224**
- Label encoding:
  - Convert breed labels into **one-hot vectors** aligned to the sorted list of unique breeds
- Data pipeline built with **tf.data**:
  - Shuffle (train only), map preprocessing, batch, and prefetch for GPU efficiency
- Training setup:
  - Optimizer: **Adam**
  - Loss: **Categorical Cross-Entropy**
  - Metrics: **Accuracy**
  - Callbacks: **EarlyStopping** + **TensorBoard** logging

## Model
A simple transfer learning architecture:
- `hub.KerasLayer(MobileNetV2)` (feature extractor)
- `Dense(120, activation="softmax")` (breed probability output)

## Evaluation
Because Kaggle does not provide labels for the test set, evaluation was done by:
- Creating a local **train/validation split**
- Tracking validation accuracy + loss
- Using TensorBoard to monitor training dynamics and identify overfitting

## Kaggle Submission
Predictions were generated on the Kaggle test set and formatted to match the required submission schema:
- Column `id` (image ID extracted from filename)
- One column per breed (120 columns)
- Each row contains **probability scores that sum to 1**

The final submission file was exported as a CSV and uploaded to Kaggle for external benchmarking.

## Custom Image Inference
The trained model was also used to predict breeds on personal dog photos by:
- Creating a test-style tf.data batch from custom image paths
- Running model inference
- Mapping prediction vectors to breed names via `argmax`

## Tech Stack
- Python
- TensorFlow 2 / Keras
- TensorFlow Hub
- tf.data pipelines
- NumPy, Pandas, Matplotlib
- Scikit-learn (train/validation split)
- Kaggle (dataset + evaluation)

## Output Artifacts
- Saved trained models (Keras format)
- TensorBoard logs
- Kaggle submission CSV containing breed probability predictions

---
**Goal:** Demonstrate an end-to-end deep learning workflow: data ingestion → preprocessing → training → evaluation → Kaggle submission → inference on new images.
