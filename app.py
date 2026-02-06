from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load artifacts
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Features scaler was trained on
SCALER_FEATURES = scaler.feature_names_in_

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":

        # ----------------------------
        # Raw input
        # ----------------------------
        input_data = {
            "tenure": float(request.form["tenure"]),
            "MonthlyCharges": float(request.form["monthly_charges"]),
            "TotalCharges": float(request.form["total_charges"]),
            "Contract": request.form["contract"],
            "InternetService": request.form["internet_service"],
            "PaymentMethod": request.form["payment_method"],

            # ✅ Missing training features → default values
            "OnlineSecurity": "No",
            "SIM_Operator": "Unknown",
            "Tenure_Quarter": "Q1"
        }

        df = pd.DataFrame([input_data])

        # One-hot encoding
        df = pd.get_dummies(df)

        # Align with MODEL features
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

        # ----------------------------
        # SAFE SCALING
        # ----------------------------
        scale_cols = [c for c in SCALER_FEATURES if c in df.columns]
        df[scale_cols] = scaler.transform(df[scale_cols])

        # Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)
