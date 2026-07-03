# 🌱 OptiCrop – Smart Agricultural Production Optimization Engine

OptiCrop is a Machine Learning-powered agricultural decision support system that recommends the most suitable crop based on soil nutrients and environmental conditions. The application also evaluates whether a farmer's planned crop is the optimal choice and provides a comprehensive research analytics dashboard for agricultural datasets.

The project combines Machine Learning, Data Analytics, and a Flask-based web application to assist farmers, researchers, and agricultural organizations in making data-driven farming decisions.

---

# Features

## 🌾 Crop Recommendation
Predicts the most suitable crop using:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

using a trained Random Forest Machine Learning model.

---

## ✅ Crop Suitability Evaluation

Allows farmers to enter the crop they are planning to cultivate.

The system compares:

- User selected crop
- Machine Learning recommended crop

and determines whether the selected crop is the optimal choice under the given environmental conditions.

---

## 📊 Research Analytics Dashboard

The analytics module performs automatic dataset analysis.

Features include:

- Dataset validation
- Missing value analysis
- Statistical summary
- Correlation analysis
- Feature distribution
- PCA
- K-Means Clustering
- ANOVA
- Mutual Information Analysis
- Feature Importance
- Dataset Quality Assessment
- Interactive Visualizations
- Research Report Generation

Supports both:

- User uploaded datasets (.csv/.xlsx)
- Built-in sample dataset

---

# Project Objectives

The system aims to

- Help farmers select the best crop
- Reduce crop failure risk
- Support agricultural researchers
- Provide automated dataset analytics
- Deliver fast ML predictions through a web application

---

# Technology Stack

## Backend

- Python
- Flask
- Flask-CORS

## Machine Learning

- Scikit-Learn
- Random Forest Classifier
- Pandas
- NumPy
- Joblib

## Frontend

- HTML5
- CSS3
- JavaScript

## Data Visualization

- Matplotlib
- Plotly
- Seaborn (Research Analytics)

---

# Machine Learning Model

### Dataset

Crop Recommendation Dataset

Input Features

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH
- Rainfall

Output

Crop Recommendation

Multiple Machine Learning algorithms were evaluated during development.

Models Tested

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Random Forest
- Gradient Boosting
- Extra Trees
- Gaussian Naive Bayes
- Support Vector Machine

Final Selected Model

**Random Forest Classifier**

Model Performance

| Metric | Value |
|---------|---------|
| Accuracy | 99.32% |
| Precision | 99.35% |
| Recall | 99.32% |
| F1 Score | 99.32% |

The Random Forest model was selected as the final deployment model due to its superior prediction accuracy and overall performance. :contentReference[oaicite:0]{index=0}

---

# Project Structure

```
OptiCrop/
│
├── app.py
├── opticrop_research_analytics_endpoint.py
│
├── models/
│   ├── random_forest_crop_recommendation.pkl
│   └── label_encoder.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   ├── images/
│   └── sample_data/
│       └── Crop_recommendation.csv
│
├── requirements.txt
└── README.md
```

---

# API Endpoints

---

## Home

```
GET /
```

Returns the OptiCrop dashboard.

---

## Crop Recommendation

```
POST /predict
```

### Request

```json
{
    "N":90,
    "P":42,
    "K":43,
    "temperature":20.8,
    "humidity":82,
    "ph":6.5,
    "rainfall":202.9
}
```

### Response

```json
{
    "success":true,
    "predicted_crop":"rice"
}
```

---

## Crop Suitability Validation

```
POST /validate-crop
```

### Request

```json
{
    "crop":"cotton",
    "N":90,
    "P":42,
    "K":43,
    "temperature":20.8,
    "humidity":82,
    "ph":6.5,
    "rainfall":202.9
}
```

### Response

```json
{
    "success":true,
    "planned_crop":"cotton",
    "recommended_crop":"rice",
    "is_best_choice":false,
    "message":"cotton is not the optimal crop. We recommend growing rice instead."
}
```

---

## Research Analytics

```
POST /research-analytics
```

Supports

- Uploaded CSV dataset
- Uploaded Excel dataset
- Built-in sample dataset

Returns a complete research analytics report including visualizations and statistical analysis.

---

# Installation

Clone the repository

```bash
git clone https://github.com/mani9441/OptiCrop.git
```

Navigate into the project

```bash
cd OptiCrop
```

Create a virtual environment

Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

The application will start at

```
http://127.0.0.1:7000
```

---

# How It Works

1. User enters soil parameters.
2. Flask receives the request.
3. Features are converted into a DataFrame.
4. The trained Random Forest model predicts the most suitable crop.
5. Label Encoder converts the predicted class back to the crop name.
6. The recommendation is returned to the frontend.
7. Users can optionally validate a planned crop against the model's recommendation.
8. Researchers can upload datasets for automated analytics and visualization.

---

# Research Analytics Workflow

```
Dataset Upload
        │
        ▼
Data Validation
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Statistical Analysis
        │
        ▼
Machine Learning Analysis
        │
        ▼
Visualization
        │
        ▼
Research Report Generation
```

---

# Future Enhancements

- Weather API integration
- Soil image analysis
- Fertilizer recommendation
- Disease detection
- Yield prediction
- Multi-language support
- Cloud deployment
- Mobile application
- GIS and satellite data integration
- Explainable AI (XAI) recommendations

---

# Testing

The application successfully passed functional, performance, and user acceptance testing.

Highlights include:

- Input validation
- Crop prediction
- Crop suitability evaluation
- API testing
- Frontend integration
- Dataset analysis
- Performance testing
- End-to-end workflow validation

Average prediction response time:

```
0.4–0.8 seconds
```

The system achieved 99.32% prediction accuracy and demonstrated stable performance under repeated prediction requests. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

---

# License

This project is developed for academic and educational purposes.

---

# Authors

**Manikanta Kalyanam**

**Mannava Sri Teja**

---

# Acknowledgements

- Scikit-Learn
- Flask
- Pandas
- NumPy
- Joblib
- Crop Recommendation Dataset (Kaggle)

---

# About OptiCrop

OptiCrop is designed to bridge the gap between agriculture and artificial intelligence by combining machine learning, statistical analytics, and an intuitive web interface. It enables farmers to make informed crop selection decisions while providing researchers with tools to analyze agricultural datasets, promoting data-driven and sustainable farming practices.