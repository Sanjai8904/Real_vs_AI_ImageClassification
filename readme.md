# Real vs AI-Generated Image Classification

## Overview
A machine learning project that classifies images as **Real** or **AI-Generated** using handcrafted image features and traditional machine learning algorithms.

The project uses statistical, edge-based, and texture-based features to identify differences between real and AI-generated images.

## Technologies Used
- Python
- NumPy, Pandas
- Scikit-learn
- Scikit-image
- SciPy
- Matplotlib, Seaborn
- Pillow
- Joblib
- Streamlit (for the web application)

## Feature Extraction
Extracted 17 image features:

- RGB channel statistics
- Grayscale statistics
- Entropy, skewness, and kurtosis
- Edge density
- GLCM texture features (contrast, homogeneity, energy, correlation)

## Machine Learning Models
- Logistic Regression
- Decision Tree
- Random Forest Classifier

The best model is selected based on validation F1-score and evaluated on the test dataset.

**Saved model:** `models/best_model.joblib`

## Project Structure

```text
real-vs-ai-classification/
├── dataset/
│   └── image_features.csv
├── graphs/
│   ├── dataset_samples.png
│   ├── final_confusion_matrix.png
│   ├── model_comparison.png
│   └── validation_model_comparison.csv
├── models/
│   └── best_model.joblib
├── src/
│   └── feature_extraction.py
├── data_prepare.py
├── download_dataset.py
├── model_training.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Execution

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/real-vs-ai-classification.git
cd real-vs-ai-classification
```

### 2. Create Virtual Environment
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Model Training
```bash
python model_training.py
```

### 5. Run Web Application
```bash
streamlit run app.py
```

## Key Learnings
- Image preprocessing and feature extraction
- Statistical and texture-based image analysis
- Machine learning classification
- Model comparison and evaluation
- Confusion matrix and classification metrics

## Disclaimer
This project is developed for educational and research purposes. Predictions are model-based estimates and are not guaranteed to identify every AI-generated image correctly.
