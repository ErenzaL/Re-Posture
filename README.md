# Re-Posture
AI-Based Posture Analysis and Alert System


## 1. Project Overview
RE:Posture is a real-time AI deployment project designed to monitor and evaluate user posture during prolonged computer use.

The project was initiated to address a common but overlooked issue: posture degradation over time. While many individuals are aware of correct sitting posture, most fail to recognize when their posture gradually deteriorates during focused work. Existing solutions either require wearable devices or provide static posture correction guidance, lacking continuous behavioral monitoring.
To solve this problem, RE:Posture introduces a non-intrusive, webcam-based system that continuously analyzes posture in real time and triggers corrective feedback when bad posture persists beyond a configurable threshold.

Technically, the system abstracts raw RGB images into skeletal keypoint representations using MediaPipe Pose, reducing environmental noise and improving generalization. A ResNet50-based transfer learning model then classifies posture states (Good / Bad) from structured skeleton images.

---
## 2. Data Collection & Preprocessing

### Dataset Summary

- Participants: 8
- Total Images: 2,383
- Good Posture: 1,026
- Bad Posture: 1,357

Images were collected using a standard 640x480 laptop webcam in diverse lighting and background conditions to improve generalization.

### Privacy & Noise Reduction Strategy

Instead of training on raw images, the pipeline:

1. Extracts 33 pose landmarks using MediaPipe Pose
2. Selects 13 upper-body keypoints
3. Renders skeleton representation on white background
4. Resizes to 224×224

This abstraction:

- Removes background bias
- Reduces clothing dependency
- Focuses on structural posture geometry
- Improves robustness with limited data

---

## 3. Model Design

### Base Architecture

- Model: ResNet50 (ImageNet Pretrained)
- Transfer Learning Applied
- Convolutional base frozen initially
- Custom classifier head added

### Output Configuration

- Output Layer: Dense(1)
- Activation: Sigmoid
- Loss Function: Binary Crossentropy
- Optimizer: Adam

### Regularization Strategy

- EarlyStopping (monitor='val_loss', patience=5)
- Restore best weights enabled
- Total epochs configured: 100
- Actual training stopped at 18 epochs

---

## 4. Model Performance

- Test Accuracy: **91.82%**
- Test Loss: 0.1834

Despite subtle visual differences between skeleton patterns, the model achieved over 0.91 accuracy, demonstrating effective posture structure learning.

Overfitting was controlled using EarlyStopping and transfer learning.

---

## 5. Real-Time Monitoring Logic

The application continuously:

1. Captures webcam frames
2. Extracts pose landmarks
3. Generates skeleton image
4. Runs inference
5. Updates UI status
6. Tracks posture duration

If bad posture persists longer than a configurable threshold (default: 300 seconds):

- Windows notification is triggered
- Snapshot is captured
- Event is logged in system history
- Thumbnail is added to gallery

---
## 6. User Interface Features

- Real-time skeleton overlay
- Confidence score display
- Posture duration tracking (Good / Bad time)
- Automatic snapshot capture
- Snapshot review mode
- Mini monitoring mode
- Configurable warning threshold
- Always-on-top mini window option
- Session statistics summary

The UI was designed to minimize distraction while maintaining awareness.

---

## 7. Installation

```bash
pip install -r requirements.txt
python -m app.main 
```
---

## 8. System Architecture
            ┌──────────────────────────┐
            │        Webcam Input       │
            └─────────────┬────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │   Video Thread Module    │
            │  (OpenCV Frame Capture)  │
            └─────────────┬────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │   Pose Estimator Module  │
            │  - MediaPipe Landmark    │
            │  - Skeleton Rendering    │
            │  - 224x224 Preprocessing │
            └─────────────┬────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │  ResNet50 Classifier     │
            │  (Transfer Learning)     │
            └─────────────┬────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │     UI Controller        │
            │  - Status Update         │
            │  - Confidence Display    │
            │  - Duration Tracking     │
            │  - Warning Trigger       │
            └─────────────┬────────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐
    │ Snapshot Logging │      │ Windows Alert     │
    │ (app_data_save)  │      │ Notification      │
    └──────────────────┘      └──────────────────┘



