from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# -------------------------------------------------------
# Initialize Flask app
# -------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------
# Load model and label encoders
# -------------------------------------------------------
model = joblib.load('random_forest_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# -------------------------------------------------------
# Home route
# -------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -------------------------------------------------------
# Predict route
# -------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Collect input data
        city = request.form.get('city')
        crime_description = request.form.get('crime_description', 'Unknown')
        day_of_week = request.form.get('day_of_week', datetime.now().strftime('%A'))
        month = int(request.form.get('month'))
        hour = int(request.form.get('hour'))

        # Step 2: Create a DataFrame with same feature names as training
        input_data = pd.DataFrame({
            'City': [city],
            'Hour': [hour],
            'DayOfWeek': [day_of_week],
            'Month': [month],
            'Crime_Description': [crime_description]
        })

        # Step 3: Encode categorical columns using saved label encoders
        for col in ['City', 'Crime_Description', 'DayOfWeek']:
            if col in label_encoders:
                le = label_encoders[col]
                # Handle unseen labels gracefully
                input_data[col] = input_data[col].apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                input_data[col] = le.transform(input_data[col])
            else:
                # If encoder missing, assign 0 (safe default)
                input_data[col] = 0

        # Step 4: Reorder columns to match training order
        expected_cols = ['City', 'Hour', 'DayOfWeek', 'Month', 'Crime_Description']
        input_data = input_data.reindex(columns=expected_cols)

        # Optional debug
        print("Model expects:", getattr(model, 'feature_names_in_', expected_cols))
        print("Input columns:", list(input_data.columns))
        print("Input data:", input_data)

        # Step 5: Predict
        prediction = model.predict(input_data)[0]
        risk_label = "High Risk" if prediction == 1 else "Low Risk"

        # Step 6: Generate reason/explanation
        if risk_label == "High Risk":
            reason = (
                f"The area '{city}' shows higher incidents of '{crime_description}' "
                f"on {day_of_week}s around {hour}:00 hours. Stay cautious."
            )
        else:
            reason = (
                f"The area '{city}' usually shows low crime activity for "
                f"'{crime_description}' on {day_of_week}s at {hour}:00 hours."
            )

        # Step 7: Render results on the webpage
        return render_template(
            'index.html',
            prediction_text=f"Predicted Risk: {risk_label}",
            reason_text=reason
        )

    except Exception as e:
        # Step 8: Handle errors gracefully
        return render_template(
            'index.html',
            prediction_text="Error in prediction",
            reason_text=f"Something went wrong: {e}"
        )

# -------------------------------------------------------
# Run Flask app
# -------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
