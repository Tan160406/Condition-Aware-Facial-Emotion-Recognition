# Condition-Aware Facial Emotion Recognition Using Adaptive Image Processing

## Project Overview
An academic computer vision project implementing condition-aware Facial Emotion Recognition (FER) using face detection, image-quality assessment, adaptive image processing and a ResNet18-based classifier.

## Emotion Classes
- Anger
- Fear
- Happiness
- Neutral
- Sadness
- Surprise

## System Workflow
1. Facial image input
2. Face detection and cropping
3. Image-condition assessment
4. Adaptive preprocessing
5. ResNet18 emotion classification
6. Baseline and adaptive prediction comparison
7. Experimental report generation

## Model Details
| Parameter | Value |
|---|---|
| Architecture | ResNet18 |
| Version | V8 |
| Input resolution | 112 x 112 RGB |
| Source image resolution | 48 x 48 |
| Classes | 6 |
| Best checkpoint epoch | 10 |
| Random seed | 42 |

## Repository Structure
```text
.
├── app_calibrated_v8_final_styled.py
├── blaze_face_short_range.tflite
├── requirements.txt
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
```bash
git clone https://github.com/Tan160406/Condition-Aware-Facial-Emotion-Recognition.git
cd Condition-Aware-Facial-Emotion-Recognition
pip install -r requirements.txt
```

## Run the Application
```bash
streamlit run app_calibrated_v8_final_styled.py
```

## Experimental Evaluation
The Dataset directory contains baseline, adaptive-processing and condition-detection results across seven image conditions.

The largest reported improvements are +9.96 percentage points in accuracy and +6.99 percentage points in Macro-F1 under low-light noise conditions. Adaptive preprocessing does not improve every condition or metric.

## Training and Data
The scripts directory contains the V8 training and condition-generation scripts. Original FER dataset images are not included; obtain them from their respective sources and follow their usage terms.

## Limitations
- Condition assessment uses heuristic measurements.
- Adaptive preprocessing has mixed results across conditions.
- Predictions represent visible facial expression categories, not a persons actual internal emotional state.

## Authors
Tanishka and Bhoomi Meena

## Run Locally

Start the Streamlit application using:

```bash
streamlit run app_calibrated_v8_final_styled.py
