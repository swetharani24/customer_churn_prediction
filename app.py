# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# ------------- Load model and scaler -------------
# Make sure churn_model.pkl and scaler.pkl are in the root of the repo
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

# ------------- Home route -------------
@app.route('/')
def home():
    return render_template('index.html')

# ------------- Predict route -------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model or not scaler:
            return "Model or Scaler not loaded. Check server logs."

        # --- Get inputs from form ---
        contract = request.form['contract']
        tenure = float(request.form['tenure'])
        sim_operator = request.form['sim_operator']
        online_security = request.form['online_security']
        tenure_quarter = float(request.form['tenure_quarter'])

        # --- Encode categorical features ---
        contract_map = {"Month-to-month":0, "One year":1, "Two year":2}
        sim_operator_map = {"OperatorA":0, "OperatorB":1, "OperatorC":2}  # update with your dataset
        online_security_map = {"No":0, "Yes":1, "No internet service":2}

        contract_encoded = contract_map[contract]
        sim_operator_encoded = sim_operator_map[sim_operator]
        online_security_encoded = online_security_map[online_security]

        # --- Prepare feature array ---
        features = np.array([[contract_encoded, tenure, sim_operator_encoded, online_security_encoded, tenure_quarter]])
        features_scaled = scaler.transform(features)

        # --- Predict ---
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]  # Probability of churn

        result = "Customer will churn 🔴" if prediction == 1 else "Customer will stay 🟢"
        prob_text = f"Churn Probability: {probability*100:.2f}%"

        return render_template('index.html', prediction_text=result, probability_text=prob_text)

    except Exception as e:
        return f"Error: {e}"

# ------------- Run app on Render -------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port)
