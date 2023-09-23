# 1. Library imports

import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle
from data_validater import DataTypes
from predict_pipeline import predict_price


# 2. Create the app object
application = FastAPI()
pickle_in = open("artifacts\pipeline.pkl","rb")
regressor=pickle.load(pickle_in)

# 3. Index route
@application.get('/')
def index():
    return {'message': 'Welcome Housing Price Prediction App'}

# 4. Make prediction
@application.post('/predict')
def predict(data:DataTypes):
    prediction = predict_price(data, regressor)
    return {
        'Expected price of this house is': prediction
    }
    
# 5. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(application, host='0.0.0.0', port=8080)
    
#uvicorn application:application --reload