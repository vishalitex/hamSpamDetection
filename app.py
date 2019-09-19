# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 20:13:10 2019

@author: vishal sharma
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
    
app = Flask(__name__)
model = pickle.load(open('multinomail.pkl', 'rb'))

countVector = pickle.load(open('count.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('frontPage.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    
    x = [request.form['mail']]    
    print(x)
    x_new=countVector.transform(x)
    output=model.predict(x_new)
    if(output[0]==0):
        str="ham"
    else:
        str="spam"
    return render_template('frontPage.html', prediction_text=str )


if __name__ == "__main__":
    app.run(debug=True)