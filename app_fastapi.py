import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pandas as pd
import pickle

app = FastAPI()
classifier = pickle.load(open("classifier.pkl", "rb"))


@app.get('/')
def index():
    return {'message' : 'Hello Oracle'}
    


@app.get('/{name}')
def get_name(name:str):
    return  {'Welcome to Oracle MLOPS' :  f'{name}'}

@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    print(data)
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])

    if prediction[0]>0.5:
        prediction = "Fake Note"
    else:
        prediction = "Bank Note"
    
    return {
        "prediction" : prediction
    }

if __name__=="__main__":
    uvicorn.run(app, host='127.0.0.1', port=9000)
