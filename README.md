# 🧠 Employee Attrition Prediction using Random Forest

This project aims to predict whether an employee is likely to leave a company (attrition) using a supervised machine learning model — specifically a *Random Forest Classifier*.

## 📌 Problem Statement

Employee attrition (or turnover) is a significant concern for many organizations, leading to loss of knowledge, training costs, and lowered team morale. By leveraging historical HR data, we can build a model to predict if an employee is at risk of leaving, which helps HR departments proactively take action.

---

## 🛠 Technologies Used

- *Python 3*
- *Pandas, **NumPy* – data manipulation
- *Seaborn, **Matplotlib* – visualization
- *Scikit-learn* – model building & evaluation
- *LabelEncoder* – encoding categorical variables
- *RandomForestClassifier* – predictive model
- *Google Colab / Jupyter Notebook* – development environment

---

## 📂 Dataset
- *Source*: IBM HR Analytics Employee Attrition & Performance
- *Filename*: WA_Fn-UseC_-HR-Employee-Attrition.csv
- *Target Variable*: Attrition (1 = Yes, 0 = No)

Features include:
- Age, Department, Education, Job Role, Monthly Income, Total Working Years, Years at Company, etc.

---

## 📈 Workflow

1. *Load and preprocess the dataset*:
    - Drop constant or irrelevant columns.
    - Encode categorical variables using LabelEncoder.

2. *Exploratory Data Analysis (EDA)*:
    - Visualize class distribution and feature importance.
    - Plot confusion matrix.

3. *Train-Test Split*:
    - 80% training, 20% testing (stratified).

4. *Model Training*:
    - Random Forest Classifier with 200 trees.

5. *Evaluation*:
    - Accuracy, Classification Report, Confusion Matrix, Feature Importance.

---

## ✅ Results
 Model Accuracy: ~*84–88%* (varies slightly depending on dataset split and random seed).
- Identified top features influencing attrition.

---

## 📊 Visuals

- 📌 *Attrition Distribution*  
- 📌 *Confusion Matrix Heatmap*  
- 📌 *Top 10 Feature Importances*

---

## 💡 Future Improvements

- Use SMOTE for class imbalance handling.
- Try other classifiers (XGBoost, Logistic Regression).
- Hyperparameter tuning with GridSearchCV.
- Build a web-based HR dashboard for real-time predictions.

