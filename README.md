<<<<<<< HEAD
# Condition-Aware Facial Emotion Recognition Using Adaptive Image Processing

## Project Overview
This project implements a condition-aware Facial Emotion Recognition (FER) system. It uses face detection, image-condition assessment, adaptive image processing and a ResNet18-based emotion classification model.

The application supports six emotion classes:
=======
# Condition-Aware Facial Emotion Recognition

A computer vision research project that combines facial expression recognition with image-condition assessment and adaptive preprocessing to study how image quality affects emotion classification.

## 1. Project Overview

Facial Emotion Recognition (FER) systems can be affected by variations in image quality, including low illumination, blur, noise and partial occlusion.

This project presents a **Condition-Aware Facial Emotion Recognition System** that assesses the visual quality of an input facial image and applies condition-specific preprocessing before performing facial expression classification.

The system uses a ResNet18-based V8 facial expression model and provides a comparison between baseline and adaptive predictions.

## 2. Key Features

- Facial detection and cropping.
- Image quality assessment using brightness, sharpness and noise measurements.
- Detection of low illumination, possible blur and possible noise.
- Adaptive image preprocessing based on detected conditions.
- Six-class facial expression recognition.
- Baseline versus adaptive prediction comparison.
- Expression score distribution visualization.
- Downloadable image analysis report.
- Experimental evaluation using accuracy and macro-F1.

## 3. System Architecture

The proposed workflow consists of the following stages:

1. **Input Image:** An image containing a human face is uploaded.
2. **Face Detection:** The face is detected and cropped for analysis.
3. **Condition Assessment:** Brightness, sharpness and noise are measured.
4. **Condition Identification:** The system identifies possible image-quality issues.
5. **Adaptive Preprocessing:** Appropriate image enhancement is selected based on the detected condition.
6. **Emotion Classification:** The ResNet18 V8 model predicts the facial expression.
7. **Prediction Comparison:** Baseline and adaptive predictions are compared.
8. **Report Generation:** An image analysis report is generated.

## 4. Model Details

| Parameter | Description |
|---|---|
| Architecture | ResNet18 |
| Model Version | V8 |
| Input Resolution | 112 × 112 |
| Input Format | RGB |
| Number of Classes | 6 |
| Model Checkpoint | fer_v8_best.pth |

### Facial Expression Categories

>>>>>>> origin/main
- Anger
- Fear
- Happiness
- Neutral
- Sadness
- Surprise

<<<<<<< HEAD
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
=======
## 5. Image Condition Assessment

The live application uses heuristic measurements to estimate image quality.

| Parameter | Assessment Rule |
|---|---|
| Brightness | Low illumination if below 85 |
| Sharpness | Possible blur if below 450 |
| Noise Indicator | Possible noise if above 16 |

The system applies adaptive preprocessing according to the detected image condition.

**Note:** Condition assessment is heuristic-based and is not a separately trained condition classifier. Partial occlusion is not reliably detected by the current live quality-assessment implementation.

## 6. Experimental Evaluation

The V8 experimental evaluation compares baseline and adaptive performance across seven image conditions.

### Key Observations

- The largest accuracy improvement occurs under low-light noise conditions (+9.96 percentage points).
- The largest macro-F1 improvement also occurs under low-light noise conditions (+6.99 percentage points).
- Performance changes vary across image conditions.
- Adaptive preprocessing does not consistently improve every evaluation metric or condition.

## 7. Repository Structure

```text
Condition-Aware-Facial-Emotion-Recognition/
│
├── Dataset/
│   └── fer_v8_best.pth
│
├── app_calibrated_v8_final_styled.py
├── blaze_face_short_range.tflite
├── requirements.txt
├── .gitattributes
└── README.md
```

## 8. Installation and Execution

### Prerequisites

- Python 3.10 or a compatible version.
- Git.
- Git LFS.
- pip.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Tan160406/Condition-Aware-Facial-Emotion-Recognition.git

cd Condition-Aware-Facial-Emotion-Recognition
```

### Step 2: Download the Model Checkpoint

Git LFS is used to store the trained model checkpoint.

```bash
git lfs install
git lfs pull
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

>>>>>>> origin/main
```bash
streamlit run app_calibrated_v8_final_styled.py
```

<<<<<<< HEAD
## Experimental Results
The `Dataset` directory contains the trained model checkpoint and CSV files containing baseline, adaptive-processing and condition-detection results.

## Training and Data Preparation
The `scripts` directory contains the V8 model-training and image-condition generation scripts.

The original FER dataset files are not included in this repository. Obtain the datasets from their respective official sources and follow their usage and licensing terms.

## Authors
Tanishka and Bhoomi Meena
=======
The application will open in your browser, generally at:

```text
http://localhost:8501
```

## 9. Research Limitations

- Condition assessment is heuristic-based rather than learned through a dedicated condition classifier.
- Partial facial occlusion is not reliably detected by the current live assessment.
- Softmax scores are not calibrated probabilities.
- Adaptive preprocessing produces mixed results depending on image condition and evaluation metric.
- The system predicts visible facial expression categories and does not establish a person's actual internal emotional state.
- Predictions may be inaccurate, particularly for underrepresented expression classes.

## 10. Technologies Used

- Python
- PyTorch
- ResNet18
- Streamlit
- OpenCV / Image Processing
- NumPy
- Pillow
- Plotly

## 11. Future Scope

- Develop a trained image-condition classification model.
- Improve partial occlusion detection.
- Explore more advanced adaptive enhancement techniques.
- Improve recognition performance across underrepresented expression categories.
- Evaluate robustness across additional datasets and real-world conditions.

## 12. Project Information

**Project:** Condition-Aware Facial Emotion Recognition  
**Model:** ResNet18 V8  
**Application:** FER Studio  
**Repository:** https://github.com/Tan160406/Condition-Aware-Facial-Emotion-Recognition

This project is developed as an academic computer vision research prototype.
>>>>>>> origin/main
