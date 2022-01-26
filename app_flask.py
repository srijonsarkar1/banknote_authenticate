import numpy as np
from flask import Flask, request, jsonify, render_template
from BankNotes import BankNote

import pickle

app = Flask(__name__)

model = pickle.load(open('classifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
    #return {'message' : 'Hello Oracle'}

@app.route('/predict', methods=['POST'])
def predict_banknote(data:BankNote):
    """
        for rendering result on HTML GUI
    """

    data = data.dict()
    print(data)
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    prediction = model.predict([[variance, skewness, curtosis, entropy]])

    if prediction[0]>0.5:
        prediction = "Fake Note"
    else:
        prediction = "Bank Note"

    return render_template('index.html', prediction_text = f'Specified note is a ${prediction}')


if __name__=="__main__":
    app.run(debug=True)
    #main()