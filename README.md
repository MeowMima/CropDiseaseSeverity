Crop Disease Severity Pipeline

An AI/ML-based pipeline for crop disease detection and severity analysis using computer vision.

Project Structure
CropDiseaseSeverity/
- cv_pipeline/                    # Computer vision analysis pipeline
- models/                         # Trained ML/DL models
- test_images/                    # Images used for testing
- outputs/                        # Generated analysis outputs
- app.py                          # Main application
- test_pipeline.py                # Pipeline testing
- PlantDiseaseDetectionYOLOv8 (1).ipynb   # Disease detection notebook
- Severity.ipynb                  # Disease severity analysis notebook
- .gitignore                      # Git ignored files and folders

GitHub: https://github.com/MeowMima/CropDiseaseSeverity

### Current Focus
- Crop disease detection
- Disease severity classification

## Overview
**Crop Disease Severity Pipeline** is a computer vision and deep learning-based system designed to analyze crop images for disease detection and estimate the severity of the detected disease.

- The pipeline combines object detection and image analysis techniques to identify diseased regions and derive severity levels from visual symptoms. It is designed as a modular pipeline that can be integrated into a larger agricultural monitoring or precision-farming system.

## Features

* 🌱 **Crop Disease Detection** — Identifies disease-affected regions in crop images.
* 🔍 **Computer Vision Analysis** — Processes crop images to extract disease-related visual information.
* 📊 **Disease Severity Estimation** — Classifies the detected disease based on its apparent severity.
* 🧩 **Modular Pipeline** — Separates image analysis, disease detection, and severity classification into reusable components.
* 🖼️ **Image-Based Testing** — Supports testing the pipeline using sample crop images.
* 🚀 **Application Interface** — Provides an application layer for running the analysis pipeline.
* 🧪 **Pipeline Testing** — Includes dedicated testing for validating the analysis workflow.

## Tech Stack

| Technology           | Purpose                                       |
| -------------------- | --------------------------------------------- |
| **Python**           | Core programming language                     |
| **YOLOv8**           | Object detection and disease-region detection |
| **OpenCV**           | Image processing and computer vision          |
| **NumPy**            | Numerical and array-based operations          |
| **Pandas**           | Data handling and analysis                    |
| **Pillow (PIL)**     | Image loading and manipulation                |
| **Streamlit**        | Application interface                         |
| **Jupyter Notebook** | Model development and experimentation         |

## Pipeline Architecture

```text
                 ┌─────────────────────┐
                 │     Input Image     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Image Processing  │
                 │   & Preprocessing    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Disease Detection │
                 │       YOLOv8        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Diseased Region   │
                 │      Analysis       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Severity Estimation │
                 │   & Classification  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      Results        │
                 │ Disease + Severity  │
                 └─────────────────────┘
```

### Pipeline Flow

**Input Image → Preprocessing → Disease Detection → Diseased Region Analysis → Severity Estimation → Final Result**

The architecture is designed to keep the detection and severity-analysis stages modular, making it easier to improve individual components without restructuring the complete application.

- Computer vision pipeline
- AI/ML-based crop analysis
