from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np


app = Flask(__name__)


model = load('classification_model.joblib')
gender_encoder = load('gender_encoder.joblib')
ever_married_encoder = load('evermarried_encoder.joblib')
work_type_encoder = load('worktype_encoder.joblib')
residence_encoder = load('residence_encoder.joblib')
smoking_encoder = load('smoking_encoder.joblib')
 

@app.route('/')
def index():
    
    
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    gender = request.form['gender'] 
    age = float(request.form['age']) 
    hypertension = int(request.form['hypertension'])  
    heart_disease = int(request.form['heart_disease'])  
    ever_married = request.form['ever_married']  
    work_type = request.form['work_type']  
    residence_type = request.form['residence_type']  
    avg_glucose_level = float(request.form['avg_glucose_level'])  
    bmi = float(request.form['bmi'])  
    smoking_status = request.form['smoking_status']  

  
    gender = gender_encoder.transform([gender])[0]
    ever_married = ever_married_encoder.transform([ever_married])[0]
    work_type = work_type_encoder.transform([work_type])[0]
    residence_type = residence_encoder.transform([residence_type])[0]
    smoking_status = smoking_encoder.transform([smoking_status])[0]

    user_input = np.array([[age, hypertension, heart_disease,avg_glucose_level, bmi,gender,  ever_married, work_type, residence_type,
                             smoking_status]])

    prediction = model.predict(user_input)
    result = "Stroke" if prediction[0] == 1 else "No Stroke"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
