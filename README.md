# 🧠 Teen Mental Health & Social Media Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)

A premium, interactive **Data Science & Machine Learning Web Application** investigating the impact of social media and lifestyle choices on adolescent mental health (ages 13–19). Built with **Streamlit**, this project performs end-to-end data analysis and trains several machine learning classifiers to predict depression risk.

---

## ✨ Features

- **🏠 Overview & KPIs:** High-level summary of the dataset (1,200 records), demographic breakdowns, interactive feature histograms, and a correlation heatmap.
- **📊 Exploratory Data Analysis (EDA):** Deep dive into the data using Box Plots, Scatter Plots, Age-based Aggregations, and Statistical Summaries to spot trends and relationships.
- **🤖 AI Models Comparison:** Interactive training and evaluation of 4 powerful classifiers (Logistic Regression, Random Forest, SVM, Gradient Boosting) using techniques like `SMOTE` (Synthetic Minority Over-sampling) and hyperparameter tuning to ensure robust, leak-free evaluation.
- **🔮 Depression Risk Predictor:** Make real-time predictions! Input a student's demographic, lifestyle, and mental wellness details into an easy-to-use form to receive a risk probability estimate alongside feature importances.
- **🎨 Custom UI/UX:** A highly customized dark mode CSS theme with glassmorphism elements, micro-animations, and Plotly interactive visualizations.

---

## 🛠️ Technology Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** Pandas, NumPy
- **Visualizations:** Plotly Express, Plotly Graph Objects
- **Machine Learning Pipeline:** Scikit-Learn
- **Class Imbalance Handling:** Imbalanced-Learn (SMOTE)

---

## 🚀 Getting Started

### 1. Requirements
Ensure you have Python 3.9+ installed on your system.

### 2. Clone the Repository
```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-DIR>
```

### 3. Install Dependencies
You can install all necessary packages via `pip`:
```bash
pip install streamlit pandas numpy plotly scikit-learn imbalanced-learn
```

### 4. Run the Application
Launch the Streamlit app from your terminal:
```bash
streamlit run app.py
```
This will automatically open the dashboard in your default web browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
├── app.py                            # Main Streamlit application and dashboard code
├── Teen_Mental_Health_Dataset.csv    # Source dataset with 1.2K adolescent records
├── mental_health_complete.ipynb      # Complete ML Notebook (EDA, Modeling, Evaluation)
├── .gitignore                        # Standard Python Git ignores
└── README.md                         # Project documentation
```

---

## 🧠 The Machine Learning Pipeline
The pipeline effectively tackles **severe class imbalance** (approx. 97.4% vs 2.6%) through the following steps:
1. Label Encoding & Dummy Variables (One-Hot)
2. Predictor Scaling using `StandardScaler`
3. Managing Outliers via IQR clipping
4. **Data Leakage Prevention:** Applying an 80/20 train-test split *before* oversampling.
5. **Class Rebalancing:** Using `SMOTE` strictly on the training subset.
6. **Modeling:** Logistic Regression, Random Forest, Support Vector Machines (RBF), and Gradient Boosting.

---

> *"Social media offers connection, but understanding its balance is the key to healthy adolescent development."*
