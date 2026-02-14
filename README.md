# Student Depression Classification System – ML Assignment

---

## Project Description:

This machine learning project focuses on predicting student depression using multiple classification techniques. Six different algorithms are trained and evaluated using standard performance metrics to determine the most reliable and effective model.

The project integrates data preprocessing, comprehensive model evaluation, and an interactive Streamlit application to support early risk detection and enable timely academic and mental health intervention.

---

## Problem Statement:

The objective of this project is to predict whether a student is **depressed or not depressed** based on demographic, academic, lifestyle, and psychological attributes.

This problem is formulated as a **binary classification task**, where multiple machine learning algorithms are trained and evaluated to determine the most reliable predictive model. Early detection of depression risk can help institutions provide timely academic support and mental health intervention.


---

## Dataset Description:

### Student Depression Dataset

This project utilizes a structured dataset containing student-related information to predict depression status.

### Dataset Overview

- **Total Records:** 27,901  
- **Total Features:** 17 input features + 1 target variable  
- **Task Type:** Supervised Machine Learning (Binary Classification)  
- **Target Variable:** `Depression`  
  - `1` → Depressed  
  - `0` → Not Depressed  
- No missing values present  

---

### Feature Description

- Gender – Indicates whether the student is male or female.
- Age – Age of the student in years.
- City – City where the student resides or studies.
- Profession – Current occupation status of the individual (primarily student).
- Academic Pressure – Level of academic stress experienced by the student (0–5 scale).
- Work Pressure – Level of work-related stress experienced (0–5 scale).
- CGPA – Student’s cumulative grade point average (0–10 scale).
- Study Satisfaction – Satisfaction level with academic studies (0–5 scale).
- Job Satisfaction – Satisfaction level with job (if applicable).
- Sleep Duration – Average daily sleep duration category.
- Dietary Habits – Quality of eating habits (Healthy/Moderate/Unhealthy).
- Degree – Academic degree program pursued by the student.
- Suicidal Thoughts – Indicates whether the student has experienced suicidal thoughts (Yes/No).
- Work/Study Hours – Average number of hours spent studying or working per day.
- Financial Stress – Level of financial pressure experienced by the student.


---

## Data Preprocessing

- Removed unnecessary ID column  
- Encoded categorical features  
- Applied feature scaling (StandardScaler)  
- Split dataset into Training (80%) and Testing (20%) sets  
- Evaluated models on unseen test data  

---

## Models Implemented

In this project, six different machine learning classification algorithms were implemented and compared to predict whether a student is depressed or not. Each model was trained and evaluated using standard performance metrics to determine the most accurate and reliable approach for depression prediction.

- Logistic Regression – A linear classification algorithm that predicts probabilities using the sigmoid function.
- Decision Tree Classifier – A tree-based model that splits data into branches based on feature thresholds.
- k-Nearest Neighbors (kNN) – A distance-based algorithm that classifies data based on the majority class of nearest neighbors. 
- Naive Bayes (GaussianNB) – A probabilistic classifier based on Bayes’ Theorem assuming feature independence. 
- Random Forest Classifier – An ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting. 
- XGBoost Classifier – A gradient boosting algorithm that builds sequential trees to enhance predictive performance. 
---

## Evaluation Metrics

| Model Name          | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
|---------------------|----------|-------|-----------|--------|----------|-------|
| Logistic Regression | 0.837    | 0.913 | 0.850     | 0.872  | 0.861    | 0.663 |
| Decision Tree       | 0.744    | 0.739 | 0.783     | 0.772  | 0.778    | 0.476 |
| KNN                 | 0.806    | 0.863 | 0.818     | 0.856  | 0.836    | 0.599 |
| Naive Bayes         | 0.827    | 0.909 | 0.864     | 0.832  | 0.848    | 0.647 |
| Random Forest       | 0.826    | 0.905 | 0.840     | 0.866  | 0.853    | 0.642 |
| XGBoost             | 0.825    | 0.900 | 0.840     | 0.862  | 0.851    | 0.638 |

---

## Model Performance Insights:

| ML Model Name            | Observation about model performance                                                                                                                              |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Performed strongly with high Accuracy (83.7%), AUC (0.913), and MCC (0.663), indicating good linear separability and balanced predictions.                       |
| Decision Tree            | Showed comparatively lower performance with Accuracy (74.4%) and MCC (0.476). Likely prone to overfitting, which reduced generalization ability.                                     |
| kNN                      | Achieved good Accuracy (80.6%) and F1-score (0.836). Performance benefited from feature scaling but remains sensitive to dataset size and noise.                      |
| Naive Bayes              | Delivered strong Precision (0.864) and competitive AUC (0.909). Assumption of feature independence may limit performance in complex relationships.                     |
| Random Forest (Ensemble) | Provided robust and consistent results with high Recall (0.866) and balanced MCC (0.642). Ensemble learning helped reduce overfitting.                       |
| XGBoost (Ensemble)       | Maintained stable and competitive performance with strong F1-score (0.851) and MCC (0.638). Gradient boosting effectively captured complex feature interactions. |

---

