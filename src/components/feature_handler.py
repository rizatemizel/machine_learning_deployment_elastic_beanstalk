import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce



class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, add_attributes=True):
        
        self.add_attributes = add_attributes
        
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X):
        
        if self.add_attributes:
            X_copy = X.copy()
            X_copy["Lack_of_feature_index"] = X_copy[["Street", "Alley", "MasVnrType", "GarageType", "MiscFeature",  'BsmtQual',
                                              'FireplaceQu','PoolQC','Fence']].isnull().sum(axis=1) + (X_copy["MasVnrType"] == 'None')+ (X_copy["CentralAir"] == 'No')
            X_copy["MiscFeatureExtended"] = (X_copy["PoolQC"].notnull()*1 + X_copy["MiscFeature"].notnull()*1+ X_copy["Fence"].notnull()*1).astype('int64')
            X_copy["Has_Alley"] = X_copy["Alley"].notnull().astype('int64')
            X_copy["Lot_occupation"] = X_copy["GrLivArea"]  / X_copy["LotArea"]
            X_copy["Number_of_floors"] = (X_copy["TotalBsmtSF"] != 0).astype('int64') + (X_copy["1stFlrSF"] != 0).astype('int64') + (X_copy["2ndFlrSF"] != 0).astype('int64')
            X_copy['Total_Close_Live_Area'] = X_copy['GrLivArea'] + X_copy['TotalBsmtSF'] 
            X_copy['Outside_live_area'] =  X_copy['WoodDeckSF'] + X_copy['OpenPorchSF'] + X_copy['EnclosedPorch']+ X_copy['3SsnPorch'] + X_copy['ScreenPorch']
            X_copy['Total_usable_area'] = X_copy['Total_Close_Live_Area'] + X_copy['Outside_live_area']
            X_copy['Area_Quality_Indicator'] = X_copy['Total_usable_area'] * X_copy['OverallQual']
            X_copy['Area_Qual_Cond_Indicator'] = X_copy['Total_usable_area'] * X_copy['OverallQual']* X_copy['OverallCond']
            X_copy['TotalBath'] = (X_copy['FullBath'] + (0.5 * X_copy['HalfBath']) + X_copy['BsmtFullBath'] + (0.5 * X_copy['BsmtHalfBath']))
            X_copy["Has_garage"] = X_copy["GarageYrBlt"].notnull().astype('int64')
            X_copy['House_Age'] = X_copy['YrSold'] - X_copy['YearBuilt']
            X_copy["Is_Remodeled"] = (X_copy["YearBuilt"] != X_copy["YearRemodAdd"]).astype('int64')
            X_copy['HasBsmt'] = X_copy['BsmtQual'].notnull().astype('int64')
            X_copy['Quality_conditition'] = X_copy['OverallQual']* X_copy['OverallCond']
            X_copy['Quality_conditition_2'] = X_copy['OverallQual'] + X_copy['OverallCond']
            X_copy['House_Age2'] = X_copy['YrSold'] - X_copy['YearRemodAdd']
            X_copy['Quality_conditition'] = X_copy['OverallQual']* X_copy['OverallCond']
            X_copy['Quality_conditition_2'] = X_copy['OverallQual'] + X_copy['OverallCond']
            X_copy['House_Age2'] = X_copy['YrSold'] - X_copy['YearRemodAdd']
            return X_copy
        else:
            return X_copy
        
        
# ordinal transformer


    

    
    