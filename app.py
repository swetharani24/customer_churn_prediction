# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- Get form inputs ---
        contract = request.form['contract']
        tenure = float(request.form['tenure'])
        sim_operator = request.form['sim_operator']
        online_security = request.form['online_security']
        tenure_quarter = float(request.form['tenure_quarter'])

        # --- Encode categorical variables ---
        contract_map = {"Month-to-month":0, "One year":1, "Two year":2}
        sim_operator_map = {"OperatorA":0, "OperatorB":1, "OperatorC":2}  # update as per dataset
        online_security_map = {"No":0, "Yes":1, "No internet service":2}

        contract_encoded = contract_map[contract]
        sim_operator_encoded = sim_operator_map[sim_operator]
        online_security_encoded = online_security_map[online_security]

        # --- Prepare features array ---
        features = np.array([[contract_encoded, tenure, sim_operator_encoded, online_security_encoded, tenure_quarter]])

        # --- Scale features ---
        features_scaled = scaler.transform(features)

        # --- Predict ---
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]  # probability of churn (class 1)

        result = "Customer will churn 🔴" if prediction == 1 else "Customer will stay 🟢"
        prob_text = f"Churn Probability: {probability*100:.2f}%"

        return render_template('index.html', prediction_text=result, probability_text=prob_text)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
