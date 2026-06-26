
🚀 Customer Churn Prediction Web App

A Machine Learning-powered web application that predicts whether a customer is likely to churn (stop using a service or product). The application analyzes customer information and returns both a Churn Prediction and a Confidence Score.

🌐 Live Demo

Try the application here:

👉 https://customer-churn-prediction-y73h.onrender.com

📖 About the Project

Customer churn prediction is the process of identifying customers who are likely to discontinue using a product or service. Accurate churn prediction enables businesses to reduce customer loss by identifying at-risk customers early and implementing effective retention strategies.

This application accepts customer information through an interactive web interface, processes the data using a trained Machine Learning model, and predicts whether the customer is likely to churn.

✨ Features
Predict customer churn in real time
User-friendly web interface
Displays prediction probability (confidence score)
Fast and accurate predictions
Machine Learning-based classification
Deployed on Render
⚙️ How It Works
Enter customer information in the web form.
The input data is preprocessed using the same techniques applied during model training.
The trained Machine Learning model analyzes the customer data.
The application displays:
Churn Prediction: Yes / No
Confidence Score: Probability of churn
📊 Model Insights

The prediction model uses customer-related information such as:

Customer tenure
Contract type
Payment method
Monthly charges
Internet service
Online security
Tech support
Device protection
Customer demographics
Service usage information
Model Output
✅ Churn Prediction (Yes / No)
📈 Confidence Score
🛠️ Tech Stack
Python
Flask
Scikit-learn
Pandas
NumPy
HTML5
CSS3
Bootstrap
Joblib
📁 Project Structure
customer_churn_prediction/
│
├── static/
│
├── templates/
│
├── model/
│   ├── model.pkl
│   ├── encoder.pkl
│   └── scaler.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
🚀 Installation

Clone the repository:

git clone https://github.com/swetharani24/customer_churn_prediction.git

Navigate to the project folder:

cd customer_churn_prediction

Install the required dependencies:

pip install -r requirements.txt

Run the Flask application:

python app.py

Open your browser and visit:

http://127.0.0.1:5000
🌍 Deployment

This application is deployed on Render.

Render automatically builds and deploys the application using the project files and requirements.txt.

📋 Usage
Open the web application.
Enter customer information.
Click the Predict button.
View the prediction result and confidence score.
📈 Business Benefits

Customer churn prediction helps organizations:

Improve customer retention
Identify high-risk customers
Reduce revenue loss
Support data-driven decision making
Increase customer lifetime value
🔮 Future Improvements
Add Explainable AI (SHAP)
Deploy using Docker
Add REST API support
Improve model accuracy with XGBoost
Store prediction history
Add user authentication
📄 License

This project is licensed under the MIT License.

⭐ Support

If you found this project useful, please consider giving it a ⭐ Star on GitHub.
