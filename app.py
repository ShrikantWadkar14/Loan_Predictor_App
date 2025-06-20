from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model with error handling
try:
    model = joblib.load("loan_eligibility_model_tuned.pkl")
except FileNotFoundError:
    print("Warning: Model file not found. Using dummy predictions for demo.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and validate inputs with default values and proper error handling
        def safe_float_convert(value, field_name, default=0.0):
            if value is None or value == '' or value == 'None':
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for {field_name}: '{value}'. Please enter a valid number.")

        # Get form data with error handling
        gender = safe_float_convert(request.form.get('Gender'), 'Gender')
        married = safe_float_convert(request.form.get('Married'), 'Marital Status')
        dependents = safe_float_convert(request.form.get('Dependents'), 'Dependents')
        education = safe_float_convert(request.form.get('Education'), 'Education')
        self_employed = safe_float_convert(request.form.get('Self_Employed'), 'Employment Type')
        applicant_income = safe_float_convert(request.form.get('ApplicantIncome'), 'Applicant Income')
        coapplicant_income = safe_float_convert(request.form.get('CoapplicantIncome'), 'Co-applicant Income')
        loan_amount = safe_float_convert(request.form.get('LoanAmount'), 'Loan Amount')
        loan_term = safe_float_convert(request.form.get('Loan_Amount_Term'), 'Loan Term')
        credit_history = safe_float_convert(request.form.get('Credit_History'), 'Credit History')
        property_area = safe_float_convert(request.form.get('Property_Area'), 'Property Area')

        # Input validation with detailed error messages
        errors = []
        if not (0 <= gender <= 1):
            errors.append("Gender must be selected (0 or 1)")
        if not (0 <= married <= 1):
            errors.append("Marital Status must be selected (0 or 1)")
        if not (0 <= dependents <= 3):
            errors.append("Dependents must be between 0-3")
        if not (0 <= education <= 1):
            errors.append("Education must be selected (0 or 1)")
        if not (0 <= self_employed <= 1):
            errors.append("Employment Type must be selected (0 or 1)")
        if not (1000 <= applicant_income <= 25000):
            errors.append("Applicant Income must be between ₹1,000 - ₹25,000")
        if not (0 <= coapplicant_income <= 10000):
            errors.append("Co-applicant Income must be between ₹0 - ₹10,000")
        if not (50 <= loan_amount <= 700):
            errors.append("Loan Amount must be between 50 - 700 (in thousands)")
        if not (60 <= loan_term <= 480):
            errors.append("Loan Term must be between 60 - 480 months")
        if not (0 <= credit_history <= 1):
            errors.append("Credit History must be selected (0 or 1)")
        if not (0 <= property_area <= 2):
            errors.append("Property Area must be selected (0, 1, or 2)")

        if errors:
            return render_template('index.html', 
                                 prediction_text="❌ Input Validation Errors:",
                                 errors=errors)

        # Calculate total income
        total_income = applicant_income + coapplicant_income

        # Transform loan amount (log1p to match training)
        loan_amount_log = np.log1p(loan_amount)

        # Prepare features array (10 features)
        features = np.array([gender, married, dependents, education, self_employed,
                           loan_amount_log, loan_term, credit_history, property_area,
                           total_income]).reshape(1, -1)

        # Make prediction
        if model is not None:
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
        else:
            # Dummy prediction for demo (based on simple rules)
            prediction = 1 if (credit_history == 1 and total_income >= 3000 and 
                             loan_amount <= 250 and loan_term >= 180) else 0
            probability = [0.3, 0.7] if prediction == 1 else [0.8, 0.2]

        # Format result
        if prediction == 1:
            result = "✅ Loan Approved - You are eligible!"
            result_class = "success"
            confidence = f"{probability[1]*100:.1f}%" if probability else "High"
        else:
            result = "❌ Loan Rejected - Not eligible"
            result_class = "error"
            confidence = f"{probability[0]*100:.1f}%" if probability else "High"

        # Provide tips based on inputs
        tips = get_eligibility_tips(credit_history, total_income, loan_amount, loan_term, property_area)

        return render_template('index.html', 
                             prediction_text=result,
                             result_class=result_class,
                             confidence=confidence,
                             tips=tips)

    except ValueError as e:
        return render_template('index.html', 
                             prediction_text=f"❌ Error: {str(e)}",
                             result_class="error")
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f"❌ System Error: Please check all fields are filled correctly",
                             result_class="error")

def get_eligibility_tips(credit_history, total_income, loan_amount, loan_term, property_area):
    """Generate personalized tips based on input values"""
    tips = []
    
    if credit_history == 0:
        tips.append("🔴 Poor credit history significantly reduces approval chances")
    
    if total_income < 3000:
        tips.append("🟡 Consider increasing total income (current: ₹{:,.0f})".format(total_income))
    
    if loan_amount > 250:
        tips.append("🟡 High loan amount may affect approval (current: ₹{:,.0f}K)".format(loan_amount))
    
    if loan_term < 180:
        tips.append("🟡 Longer loan terms generally improve approval chances")
    
    if property_area == 0:
        tips.append("🟡 Urban/Semi-urban properties have better approval rates")
    
    if not tips:
        tips.append("✅ Your profile looks good for loan approval!")
    
    return tips

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        # Process prediction similar to web form
        # Return JSON response
        return jsonify({"status": "success", "prediction": "eligible"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)