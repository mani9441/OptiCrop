
# 🌱 OptiCrop – Smart Agricultural Production Optimization Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-Web_App-000000?style=for-the-badge&logo=flask"/>
  <img src="https://img.shields.io/badge/Machine_Learning-Random_Forest-2E8B57?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-Model-F7931E?style=for-the-badge&logo=scikitlearn"/>
  <img src="https://img.shields.io/badge/License-Academic-blue?style=for-the-badge"/>
</p>

<p align="center">
A Machine Learning-powered agricultural decision support system for intelligent crop recommendation,
crop suitability evaluation, and comprehensive agricultural research analytics.
</p>

---

## Table of Contents

- Features
- Project Objectives
- Technology Stack
- Machine Learning Model
- Project Structure
- API Endpoints
- Installation
- How It Works
- Research Analytics Workflow
- Future Enhancements
- Testing
- License
- Authors
- Acknowledgements
- About OptiCrop

---

## Features

### 🌾 Crop Recommendation

Predicts the most suitable crop using:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

using a trained Random Forest Machine Learning model.

> [!NOTE]
> OptiCrop predicts the most suitable crop based on soil nutrients and environmental conditions using a trained Random Forest model.

### Crop Suitability Evaluation

Allows farmers to enter the crop they are planning to cultivate.

The system compares:

- User selected crop
- Machine Learning recommended crop

and determines whether the selected crop is the optimal choice under the given environmental conditions.

### 📊 Research Analytics Dashboard

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

> [!TIP]
> Researchers can upload their own CSV or Excel datasets for automatic exploratory data analysis and report generation.

---

## Project Objectives

The system aims to:

- Help farmers select the best crop
- Reduce crop failure risk
- Support agricultural researchers
- Provide automated dataset analytics
- Deliver fast ML predictions through a web application

---

## Technology Stack

| Category | Technologies |
|-----------|--------------|
| **Backend** | Python, Flask, Flask-CORS |
| **Machine Learning** | Scikit-Learn, Random Forest Classifier, Pandas, NumPy, Joblib |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Visualization** | Matplotlib, Plotly, Seaborn |

---

## Machine Learning Model

### Dataset

Crop Recommendation Dataset

### Input Features

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH
- Rainfall

### Output

Crop Recommendation

### Models Tested

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Random Forest
- Gradient Boosting
- Extra Trees
- Gaussian Naive Bayes
- Support Vector Machine

### Final Selected Model

**Random Forest Classifier**

### Model Performance

| Metric | Value |
|---------|------:|
| Accuracy | 99.32% |
| Precision | 99.35% |
| Recall | 99.32% |
| F1 Score | 99.32% |

The Random Forest model was selected as the final deployment model due to its superior prediction accuracy and overall performance.

---

## Project Structure

```text
OptiCrop/
│
├── app.py
├── opticrop_research_analytics_endpoint.py
├── models/
├── templates/
├── static/
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home Dashboard |
| POST | `/predict` | Crop Recommendation |
| POST | `/validate-crop` | Crop Suitability Evaluation |
| POST | `/research-analytics` | Research Analytics |

### GET /

Returns the OptiCrop dashboard.

### POST /predict

Request

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

Response

```json
{
  "success": true,
  "predicted_crop": "rice"
}
```

### POST /validate-crop

Request

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

Response

```json
{
  "success":true,
  "planned_crop":"cotton",
  "recommended_crop":"rice",
  "is_best_choice":false,
  "message":"cotton is not the optimal crop. We recommend growing rice instead."
}
```

### POST /research-analytics

Supports uploaded CSV datasets, Excel datasets, and the built-in sample dataset.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/mani9441/OptiCrop.git
```

### Navigate to Project

```bash
cd 5.Project_Development/d_development_integration
```

### Create Virtual Environment

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

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Application URL

```text
http://127.0.0.1:7000
```

---

## How It Works

1. User enters soil parameters.
2. Flask receives the request.
3. Features are converted into a DataFrame.
4. The trained Random Forest model predicts the most suitable crop.
5. Label Encoder converts the predicted class back to the crop name.
6. The recommendation is returned to the frontend.
7. Users can validate their planned crop.
8. Researchers can upload datasets for analytics.

---

## Research Analytics Workflow

```text
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

## Future Enhancements

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

## Testing

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

```text
0.4–0.8 seconds
```

The system achieved **99.32%** prediction accuracy and demonstrated stable performance under repeated prediction requests.

---

## License

This project is developed for academic and educational purposes.

---

## Authors

| <img src="https://github.com/mani9441.png" width="100" height="100"/> | <img src="https://github.com/sriteja-mannava.png" width="100" height="100"/> |
| :---: | :---: |
| <b>Manikanta Kalyanam</b><br><br>[![GitHub](https://img.shields.io/badge/GitHub-mani9441-181717?style=flat&logo=github)](https://github.com/mani9441) | <b>MANNAVA SRI TEJA</b><br><br>[![GitHub](https://img.shields.io/badge/GitHub-sriteja--mannava-181717?style=flat&logo=github)](https://github.com/sriteja-mannava) |

---

## Acknowledgements

- Scikit-Learn
- Flask
- Pandas
- NumPy
- Joblib
- Crop Recommendation Dataset (Kaggle)

---

## About OptiCrop

OptiCrop is designed to bridge the gap between agriculture and artificial intelligence by combining machine learning, statistical analytics, and an intuitive web interface. It enables farmers to make informed crop selection decisions while providing researchers with tools to analyze agricultural datasets, promoting data-driven and sustainable farming practices.
