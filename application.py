# 1. Library imports

import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle
from data_validater import DataTypes
from predict_pipeline import predict_price


# 2. Create the app object
app = FastAPI()
pickle_in = open("artifacts\pipeline.pkl","rb")
regressor=pickle.load(pickle_in)

# 3. Index route
@app.get('/')
def index():
    return {'message': 'Welcome Housing Price Prediction App'}

# 4. Make prediction
@app.post('/predict')
def predict(data:DataTypes):
    prediction = predict_price(data, regressor)
    return {
        'Expected price of this house is': prediction
    }
    
# 5. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn main:app --reload