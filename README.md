# Disease Diagnose Prediction

This project develops a machine learning pipeline to predict diabetes diagnoses using the PIMA Indians Diabetes dataset. The goal is to identify key factors associated with diabetes and build predictive models to aid in early diagnosis.

Data Loading and Exploration: Loads the PIMA Indians Diabetes dataset and performs initial analysis to understand feature distributions and patterns.
Data Preprocessing: Handles missing values, scales numerical features using StandardScaler, and prepares data for modeling.
Exploratory Data Analysis (EDA): Visualizes feature correlations and distributions using histograms, box plots, and correlation heatmaps.
Feature Selection and Model Training: Selects relevant features and trains Gradient Boosting and SVM models for diabetes prediction.
Performance Evaluation: Assesses model performance using Accuracy, Precision, Recall, F1 Score, and ROC AUC scores.

## 🔍 Dataset

Source: PIMA Indians Diabetes Dataset from the UCI Machine Learning Repository.
Description: Contains diagnostic measurements (e.g., glucose, blood pressure, BMI) for female patients of PIMA Indian heritage, with a binary target indicating diabetes diagnosis (0 = No, 1 = Yes).

## 🛠️ Requirements
To run the notebook, install the following Python libraries:
pip install pandas numpy matplotlib seaborn scikit-learn

## 🚀 How to Run

1. Clone the Repository (if applicable, or skip to step 2):
``` bash
git clone https://github.com/your-username/disease-diagnosis-prediction.git
cd disease-diagnosis-prediction
```

2. Install Required Libraries:
- pip install pandas numpy matplotlib seaborn scikit-learn


3. Run the Notebook:

- Download the PIMA Indians Diabetes dataset from Kaggle and place it in your working directory (e.g., as diabetes.csv).
- Open task5.ipynb in Jupyter Notebook or JupyterLab and run all cells.



📈 Final Results
Model Performance:

Gradient Boosting Classifier:

Class 0 (Non-diabetic):
Precision: 0.80
Recall: 0.84
F1 Score: 0.82
Support: 100


Class 1 (Diabetic):
Precision: 0.67
Recall: 0.61
F1 Score: 0.64
Support: 54


Overall Accuracy: 76%
Macro Average:
Precision: 0.74
Recall: 0.73
F1 Score: 0.73


Weighted Average:
Precision: 0.76
Recall: 0.76
F1 Score: 0.76


AUC-ROC Score: 0.837


SVM Classifier:

Class 0 (Non-diabetic):
Precision: 0.77
Recall: 0.84
F1 Score: 0.80
Support: 100


Class 1 (Diabetic):
Precision: 0.64
Recall: 0.54
F1 Score: 0.59
Support: 54


Overall Accuracy: 73%
Macro Average:
Precision: 0.71
Recall: 0.69
F1 Score: 0.69


Weighted Average:
Precision: 0.74
Recall: 0.75
F1 Score: 0.74


AUC-ROC Score: 0.809



## Key Insight:
The Gradient Boosting model outperforms SVM with higher accuracy (76% vs. 73%) and AUC-ROC (0.837 vs. 0.809), showing better discrimination between diabetic and non-diabetic cases. However, both models have lower recall for the diabetic class, indicating potential class imbalance. Techniques like SMOTE, feature engineering, or hyperparameter tuning could improve performance on the minority class.
## 📊 Output

- Visualizations including feature distributions, correlation heatmaps, and ROC curves.
- Classification reports with precision, recall, and F1 scores for both models.
- Feature importance analysis to identify key predictors (e.g., Glucose, BMI).

## 🤝 Contributing
- Contributions are welcome! Fork the repo and submit a pull request for improvements, additional models (e.g., Random Forest, XGBoost), or enhanced visualizations.
## 📜 License
- This project is open-source and available under the MIT License.
