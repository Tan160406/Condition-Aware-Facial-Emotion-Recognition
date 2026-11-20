# Condition-Aware Facial Emotion Recognition Using Adaptive Image Processing

## Project Overview
This project implements a condition-aware Facial Emotion Recognition (FER) system. It uses face detection, image-condition assessment, adaptive image processing and a ResNet18-based emotion classification model.

The application supports six emotion classes:
- Anger
- Fear
- Happiness
- Neutral
- Sadness
- Surprise

## Model Details
- Architecture: ResNet18
- Pretrained initialization: ImageNet
- Input size: 112 × 112 RGB
- Source image size: 48 × 48
- Number of classes: 6
- Random seed: 42

## Repository Structure
```text
Condition-Aware-Facial-Emotion-Recognition/
├── app_calibrated_v8_final_styled.py
├── blaze_face_short_range.tflite
├── requirements.txt
├── README.md
├── Dataset/
│   ├── fer_v8_best.pth
│   ├── v8_baseline_condition_results.csv
│   ├── v8_adaptive_processing_results.csv
│   └── v8_condition_detection_rates.csv
└── scripts/
    ├── 29_train_fer_v8.py
    └── 13_generate_conditions.py
```

## Installation
Python 3.12 is recommended.

```bash
git clone https://github.com/Tan160406/Condition-Aware-Facial-Emotion-Recognition.git
cd Condition-Aware-Facial-Emotion-Recognition
pip install -r requirements.txt
```

## Run the Application
```bash
streamlit run app_calibrated_v8_final_styled.py
```

## Experimental Results
The `Dataset` directory contains the trained model checkpoint and CSV files containing baseline, adaptive-processing and condition-detection results.

## Training and Data Preparation
The `scripts` directory contains the V8 model-training and image-condition generation scripts.

The original FER dataset files are not included in this repository. Obtain the datasets from their respective official sources and follow their usage and licensing terms.

## Authors
Tanishka and Bhoomi Meena
