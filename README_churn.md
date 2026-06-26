<div align="center">

# 📈 Customer Churn Prediction

### A machine learning–powered web application that predicts whether a customer is likely to churn.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Model-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-Frontend-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)](https://getbootstrap.com/)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)]()

<br/>

> The model analyzes customer information and returns both a **binary outcome** and **confidence score**.

### 🔗 [View Live Demo](https://customer-churn-prediction-y73h.onrender.com)

</div>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [How It Works](#️-how-it-works)
- [Model Insights](#-model-insights)
- [Built With](#️-built-with)
- [Deployment](#-deployment)
- [Usage Instructions](#-usage-instructions)
- [Why Churn Prediction Matters](#-why-churn-prediction-matters)
- [Notes](#-notes)

---

## 🧠 About the Project

Customer churn prediction is the process of identifying which customers are likely to **stop using a product or service**. Accurate churn prediction helps businesses:

- Reduce churn costs by identifying at-risk customers **early**
- Take **proactive retention actions** before it's too late

This application takes user inputs related to customer attributes and feeds them into a pre-trained machine learning model to determine the **likelihood of churn**.

---

## ⚙️ How It Works

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│   Input Form    │ ──► │  ML Classification   │ ──► │  Churn Prediction    │
│ Customer Details│     │  Model (scikit-learn) │     │  Yes/No + Confidence │
└─────────────────┘     └──────────────────────┘     └──────────────────────┘
```

| Step | Description |
|------|-------------|
| **1. Input Form** | Users fill in customer details such as demographic and service information |
| **2. ML Model** | A classification model trained on historical churn data applies one-hot encoding for categorical features and scaling for numeric values |
| **3. Prediction** | The model outputs whether the customer is likely to churn and displays the probability |

---

## 📊 Model Insights

### Features Used

| Feature | Type |
|---------|------|
| Tenure & usage metrics | Numeric |
| Contract type | Categorical |
| Payment methods | Categorical |
| Monthly charges | Numeric |
| Customer service interactions | Categorical |

> ℹ️ The exact feature list may vary depending on the model training dataset.

### Model Output

```
✅ Churn Prediction  →  Yes / No
📊 Confidence Score  →  Probability of churn (0.0 – 1.0)
```

---

## 🛠️ Built With

| Technology | Role | Badge |
|---|---|---|
| Python | Core Language | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) |
| scikit-learn | ML Model | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) |
| Flask | Backend API | ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white) |
| HTML/CSS | Frontend | ![HTML](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white) |
| Bootstrap | UI Styling | ![Bootstrap](https://img.shields.io/badge/Bootstrap-7952B3?style=flat&logo=bootstrap&logoColor=white) |
| Render | Deployment | ![Render](https://img.shields.io/badge/Render-46E3B7?style=flat&logo=render&logoColor=white) |

---

## 🚀 Deployment

The app is deployed on **[Render](https://render.com/)** — a cloud hosting platform that automates build and deployment steps using `requirements.txt` and application scripts.

```
GitHub Repo  ──►  Render Auto-Deploy  ──►  Live Web App
```

[![Live App](https://img.shields.io/badge/🌐%20Open%20Live%20App-46E3B7?style=for-the-badge)](https://customer-churn-prediction-y73h.onrender.com)

---

## 💡 Usage Instructions

```
1️⃣  Open the live web app in your browser
       ↓
2️⃣  Fill in the required input fields (customer details)
       ↓
3️⃣  Click the Submit button
       ↓
4️⃣  View the churn prediction result and confidence score
```

---

## 📊 Why Churn Prediction Matters

> Predicting customer churn is a **crucial capability** for subscription-based and recurring-revenue businesses.

| Benefit | Impact |
|---------|--------|
| 🔍 Identify at-risk customers | Before they leave |
| 🎯 Personalized retention strategies | Higher conversion to loyalty |
| 💰 Improve customer lifetime value | Better revenue forecasting |

By anticipating churn, companies can **allocate resources more effectively** and foster long-term loyalty.

---

## 📝 Notes

> [!NOTE]
> This app is intended for **demo and educational purposes**.

> [!WARNING]
> Prediction results should be considered alongside **business context** and additional data.

> [!IMPORTANT]
> Accuracy depends on **model quality** and **feature engineering**.

---

<div align="center">

Made with ❤️ using Python & Flask &nbsp;|&nbsp; Deployed on Render

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Render](https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=white)](https://render.com/)

</div>
