# 🧠 Employee Attrition Prediction System

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-Best_Model-orange?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> An end-to-end Machine Learning system that predicts employee attrition risk using behavioral, demographic, and workplace satisfaction data — with full model explainability and a deployed interactive web application.

🔗 **[Live Demo → Streamlit App](https://employee-attrition-predictor-e8pgspzdxmu5inxxcqovce.streamlit.app/)**
&nbsp;&nbsp;&nbsp;&nbsp;📂 **Dataset:** IBM HR Analytics Employee Attrition Dataset (1,470 employees · 35 features)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Feature Importance (SHAP)](#feature-importance-shap)
- [Business Insights](#business-insights)
- [Tech Stack](#tech-stack)
- [How to Run Locally](#how-to-run-locally)
- [REST API](#rest-api)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)

---

## 🔍 Overview

Employee attrition costs companies significantly in recruitment, training, and lost productivity. This project builds a **production-ready ML pipeline** that:

- Performs **Exploratory Data Analysis (EDA)** to uncover attrition patterns
- Applies **SMOTE** to handle class imbalance (only ~16% of employees leave)
- Trains and compares **3 ML models** to identify the best performer
- Uses **SHAP** to explain why a specific employee is predicted to leave
- Deploys a **Streamlit web app** with Low / Medium / High risk prediction
- Exposes a **REST API** for programmatic access to predictions

---

## 📁 Project Structure

```
employee-attrition-predictor/
│
├── api/                    # FastAPI REST API for model serving
├── assets/                 # Screenshots and images for README
├── dashboard/              # Power BI dashboard files
├── data/                   # Raw and processed datasets
├── models/                 # Saved trained model files (.pkl)
├── notebooks/              # Jupyter notebooks (EDA + model training)
├── src/                    # Core Python scripts (preprocessing, training)
├── streamlit_app/          # Streamlit web application
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 📊 Model Performance

Three models were trained and compared. XGBoost was selected as the final model based on highest ROC-AUC score.

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 80%      | 0.61      | 0.57   | 0.59     | 0.76    |
| Random Forest       | 83%      | 0.69      | 0.62   | 0.65     | 0.81    |
| **XGBoost ✅**      | **86%**  | **0.74**  | **0.68** | **0.71** | **0.84** |

> **Why XGBoost?** It handles feature interactions and class imbalance better than Logistic Regression, and outperforms Random Forest on recall — which is critical here since missing a high-risk employee (false negative) is more costly than a false positive.

**Evaluation strategy:** Stratified 5-fold cross-validation · SMOTE applied only on training folds to prevent data leakage.

---

## 🔎 Feature Importance (SHAP)

SHAP (SHapley Additive exPlanations) was used to explain both global model behaviour and individual predictions.



**Top 5 attrition drivers identified:**

| Rank | Feature | Impact |
|------|---------|--------|
| 1 | **OverTime** | Employees working overtime are ~3x more likely to leave |
| 2 | **MonthlyIncome** | Lower income bands show significantly higher attrition |
| 3 | **JobSatisfaction** | Dissatisfied employees (score 1–2) show 2x higher risk |
| 4 | **YearsAtCompany** | Employees under 2 years are the highest risk group |
| 5 | **WorkLifeBalance** | Poor work-life balance is a leading early-exit signal |

---

## 💡 Business Insights

Analysis of the IBM HR dataset revealed the following actionable insights for HR teams:

- **Overtime is the #1 risk factor** — employees on mandatory overtime show dramatically higher attrition regardless of salary
- **Early tenure is critical** — attrition peaks in the first 2 years; onboarding and mentorship programs in this window have the highest ROI
- **Compensation thresholds matter** — attrition drops sharply above a monthly income threshold; targeted salary reviews for mid-band employees can reduce churn
- **Job satisfaction is predictive at scale** — a single-point drop in job satisfaction score correlates with measurable attrition increase across departments
- **Environmental satisfaction is underrated** — workspace and team environment satisfaction scores are stronger predictors than many demographic features

These insights can directly inform HR retention strategy, performance review cycles, and compensation benchmarking.

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.9 |
| Data Processing | Pandas, NumPy |
| ML Models | Scikit-learn, XGBoost |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web App | Streamlit |
| API | FastAPI |
| Version Control | Git, GitHub |
| Deployment | Streamlit Cloud |

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Shravani-36/employee-attrition-predictor.git
cd employee-attrition-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```bash
streamlit run streamlit_app/app.py
```

**4. Run the REST API (optional)**
```bash
cd api
uvicorn main:app --reload
```
API will be available at `http://localhost:8000`

---

## 🔌 REST API

The project includes a FastAPI REST endpoint for programmatic attrition prediction.

**Endpoint:** `POST /predict`

**Sample Request:**
```json
{
  "Age": 34,
  "OverTime": "Yes",
  "MonthlyIncome": 3500,
  "JobSatisfaction": 2,
  "YearsAtCompany": 1,
  "WorkLifeBalance": 1
}
```

**Sample Response:**
```json
{
  "attrition_risk": "High",
  "probability": 0.83,
  "top_factors": ["OverTime", "JobSatisfaction", "YearsAtCompany"]
}
```

---

## 📸 Screenshots

### Home Page
![Home Page](assets/home.png)

### Low Attrition Risk Prediction
![Low Risk](assets/low_risk.png)

### Medium Attrition Risk
![Medium Risk](assets/medium_risk.png)

### High Attrition Risk Prediction
![High Risk](assets/high_risk.png)

---

## 🚀 Future Improvements

- [ ] MLflow experiment tracking and model versioning
- [ ] Docker containerization for portable deployment
- [ ] Database integration for prediction history logging
- [ ] User authentication for HR team access control
- [ ] Real-time model retraining pipeline with new data
- [ ] Power BI dashboard integration via embedded link

---

## 👩‍💻 About

Built by **Shravani Kola** — Data Science undergraduate at Malla Reddy University (CGPA: 9.0)

[![GitHub](https://img.shields.io/badge/GitHub-Shravani--36-black?style=flat&logo=github)](https://github.com/Shravani-36)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/your-linkedin-here)

---

## 📄 License

This project is licensed under the MIT License.
