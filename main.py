# 1. Library imports
import uvicorn
from fastapi import FastAPI
from data_validater import DataTypes
import pandas as pd
import numpy as np
import pickle
from fastapi.encoders import jsonable_encoder


# 2. Create the app object
app = FastAPI()
pickle_in = open("artifacts\pipeline.pkl","rb")
regressor=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Welcome Housing Price Prediction App'}

# 4. Make prediction
@app.post('/predict')
def predict_price(data:DataTypes):
    data = data.dict()

    # Convert the dictionary into a dataframe
    my_data = pd.DataFrame([data])

    
    #prediction
    prediction = regressor.predict(my_data)
    prediction = prediction.tolist()
    return {
        'Expected price of this house is': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn main:app --reload