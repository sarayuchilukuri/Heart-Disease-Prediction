






from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/'
model = pickle.load(open("heart_disease_predictor.pkl", "rb"))

@app.route('/')
def home():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'heart_patient.jpg')
    return render_template("index.html", house_image=pic1)

@app.route("/display" , methods=['GET', 'POST'])
def uploader():    
    if request.method=='POST':
        age = int(request.form["age"]) 
        sex = int(request.form["sex"])
        cp = int(request.form["cp"])
        trestbps = int(request.form["trestbps"])
        chol = int(request.form["chol"])
        fbs = int(request.form["fbs"])
        restecg = int(request.form["restecg"])
        thalach = int(request.form["thalach"])
        exang = int(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = int(request.form["slope"])
        ca = int(request.form["ca"])
        thal = int(request.form["thal"])

        input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model.predict(input_data_reshaped)
        if (prediction[0]== 0):
            result = "Not a Heart Patient"
        else:
            result = "A Heart Patient"
        pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'heart_patient.jpg')
        return render_template("display.html", result=result, house_image=pic1)

# input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
# input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
# input_data_as_numpy_array= np.asarray(input_data)
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# prediction = model.predict(input_data_reshaped)
# if (prediction[0]== 0):
#     result = "Not a Heart Patient"
# else:
#     result = "A Heart Patient"
  

if __name__ == '__main__':
    app.run(debug=True) 