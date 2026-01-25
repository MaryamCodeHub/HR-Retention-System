# HR-Retention-System
Built a machine learning system to predict employee attrition using HR data. Performed EDA, feature engineering, and trained classification models including Random Forest. Provided actionable HR insights to improve employee retention and workforce planning.

## Project Overview
This project analyzes HR data from Salifort Motors to predict whether an employee is likely to leave the company. By using machine learning models and data-driven insights, this project helps HR teams identify key factors contributing to employee attrition and improve retention strategies.

## Objectives

- Predict whether an employee will leave the company
- Identify key factors influencing employee attrition
- Provide actionable business recommendations for HR
- Build and evaluate classification models

## Dataset

- Source: Kaggle HR Analytics Dataset
- Records: 14,999 employees
- Features include:
- Satisfaction level
- Last evaluation
- Number of projects
- Monthly working hours
- Tenure
- Department
- Salary
- Promotion history
- Work accidents

## Tools & Technologies

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## Key Steps
###  1. Data Cleaning & Preprocessing
- Removed duplicates
- Renamed columns
- Handled outliers
- Encoded categorical variables

### 2. Exploratory Data Analysis (EDA)
- Analyzed relationships between working hours, satisfaction & attrition
- Identified overworked employees
- Visualized trends using boxplots, histograms, and heatmaps

### 3. Model Building
- Implemented and compared:
- Logistic Regression
- Decision Tree
- Random Forest

Performed:
- Hyperparameter tuning
- Cross-validation
- Feature engineering (created overworked feature)

### 4. Model Evaluation
Metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- AUC

## Best Model: Random Forest
Achieved strong performance with high AUC and accuracy.

## Key Insights
- Overworked employees are more likely to leave
- High project load + long working hours reduce retention
- Employees with tenure above 6 years are less likely to quit
- Promotion and workload balance strongly affect attrition
