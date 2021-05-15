# -*- coding: utf-8 -*-
"""
html image credits:
https://blog.unpakt.com/wp-content/uploads/2017/06/63064200_l-education-graduation-and-people-concept-silhouettes-of-many-happy-students-in-gowns-throwing-mortarboards-in-air-1024x651.jpg
"""

import pandas as pd
from flask import Flask, request, render_template
import pickle


filename = '/datasets_14872_228180_Admission_Predict_Ver1.1/LinearRegression.sav'
LinearRegression = pickle.load(open(filename, 'rb'))


app = Flask('__name__')

@app.route("/")
def loadPage():
    
	return render_template('home_flask.html', query="")

@app.route("/", methods=['POST'])
def predict():

    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7]]
    features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']
    df = pd.DataFrame(data, columns = features)
    
    a = round(LinearRegression.predict(df)[0]*100,2)
    if a > 100:
        a = 100
    
    result = f'Chances of Admit is: {a} %'
    error = 'Accuracy of the prediction: 81.88% (referred from r2_score)' 
    
    return render_template('home_flask.html', output1=result, output2=error, query1 = request.form['query1'], query2 = request.form['query2'],query3 = request.form['query3'],query4 = request.form['query4'],query5 = request.form['query5'],query6 = request.form['query6'],query7 = request.form['query7'])
    

if __name__ == "__main__":
    
    app.run(debug = True)
