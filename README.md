
# ⚡ Customer Churn Predictor

## 📌 Project Overview

Customer Churn Predictor is a Machine Learning web application built using Python and Streamlit. It predicts whether a customer is likely to leave a telecom service based on important customer details such as tenure, contract type, monthly charges, total charges, internet service, and payment method.

The application uses a trained Machine Learning model to calculate the probability of customer churn and classifies customers into **High Churn Risk** or **Low Churn Risk** categories.

## 🎯 Objectives

- Predict customer churn using Machine Learning.
- Identify customers who are at high risk of leaving a service.
- Understand how customer details influence churn predictions.
- Provide an easy-to-use interface for instant predictions.

## 🛠️ Technologies Used

- **Python** – Programming language
- **Pandas** – Data manipulation and preprocessing
- **NumPy** – Numerical operations
- **Scikit-learn** – Machine Learning and feature scaling
- **Joblib** – Saving and loading trained model files
- **Streamlit** – Interactive web application

## ⚙️ Key Features

- Simple and interactive user interface.
- Customer information input through sliders, dropdowns, and number fields.
- One-hot encoding of categorical features.
- Feature scaling using a pre-trained scaler.
- Machine Learning-based churn probability prediction.
- High and Low Churn Risk classification.
- Visual probability progress indicator.

## 🔄 Project Workflow

1. User enters customer details.
2. Input data is converted into a Pandas DataFrame.
3. Categorical features are converted using one-hot encoding.
4. Input columns are aligned with the model's training columns.
5. Continuous numerical features are scaled.
6. The trained Machine Learning model predicts churn probability.
7. The application displays the prediction result.

## 📂 Project Structure

```text
Customer-Churn-Predictor/
│
├── app.py
├── churnprediction_model.pkl
├── scaler.joblib
├── model_columns.pkl
├── requirements.txt
└── README.md
```

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project Folder

```bash
cd Customer-Churn-Predictor
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Prediction Categories

| Churn Probability | Result |
|---|---|
| 50% or above | High Churn Risk |
| Below 50% | Low Churn Risk |

## 💡 Use Case

Telecommunication companies can use customer churn prediction systems to identify customers who may discontinue their services. This helps businesses take proactive steps, improve customer satisfaction, and develop customer retention strategies.


## 👩‍💻 Author

**Anamika R**

Machine Learning Project
