# 1. Library imports

from data_validater import DataTypes
import pandas as pd
import numpy as np

# 2. Prediction pipeline


def predict_price(data:DataTypes, model):
    data = data.dict()
    data = pd.DataFrame([data])
    prediction = model.predict(data)
    prediction = prediction.tolist()
    return  prediction

