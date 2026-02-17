# Re-Posture
AI-Based Posture Analysis and Alert System


## 1. Project Overview
RE:Posture is a real-time AI deployment project designed to monitor and evaluate user posture during prolonged computer use.
The project was initiated to address a common but overlooked issue: posture degradation over time. While many individuals are aware of correct sitting posture, most fail to recognize when their posture gradually deteriorates during focused work. Existing solutions either require wearable devices or provide static posture correction guidance, lacking continuous behavioral monitoring.
To solve this problem, RE:Posture introduces a non-intrusive, webcam-based system that continuously analyzes posture in real time and triggers corrective feedback when bad posture persists beyond a configurable threshold.
Technically, the system abstracts raw RGB images into skeletal keypoint representations using MediaPipe Pose, reducing environmental noise and improving generalization. A ResNet50-based transfer learning model then classifies posture states (Good / Bad) from structured skeleton images.
