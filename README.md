# Sign Language Gloss Recognition using DINOv2

## Overview
This project implements sign language gloss recognition using a pretrained DINOv2 Vision Transformer model. The system processes RGB video frames of sign gestures and predicts the corresponding gloss label.

## Dataset
LSA64 Sign Language Dataset

- 64 gloss classes
- RGB video sequences
- Frame sampling used for temporal processing

## Method
## Architecture

The system follows a video classification pipeline:

Video Input  
↓  
Frame Sampling (8 RGB Frames)  
↓  
DINOv2 Vision Transformer Feature Extractor  
↓  
Temporal Average Pooling  
↓  
Fully Connected Classification Head  
↓  
Predicted Gloss Label

Pipeline:

Video → Frame Sampling → DINOv2 Feature Extractor → Temporal Average Pooling → Classification Head → Gloss Prediction

## Model
- Backbone: DINOv2 Vision Transformer
- Framework: PyTorch
- Feature aggregation using temporal average pooling

## Results
The model learns visual representations of hand gestures and classifies sign language glosses across multiple classes.

## Tech Stack
Python  
PyTorch  
Transformers  
OpenCV  
NumPy  

## Future Work
- Add temporal models such as LSTM or Temporal Transformers
- Real-time inference system
- Sentence-level sign language translation
