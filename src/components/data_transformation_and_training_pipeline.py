import sys
from dataclasses import dataclass
import os

import numpy as np 
import pandas as pd

#preprocessing
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce


#algorithms
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklego.linear_model import LADRegression

#transformers and pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor

#model evaluation
from sklearn.metrics import mean_absolute_error

#stacking
from sklearn.ensemble import StackingRegressor

#from this package
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from feature_handler import FeatureCreator


#not for myself: next time use a config file to manage feature configuration
categorical_features = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                        'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
                        'MiscFeature', 'SaleType', 'SaleCondition']

ordinal_features = ['LotShape', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond',
                    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                    'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
                    'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']

nominal_features = ['HouseStyle', 'BldgType', 'MasVnrType', 'Electrical', 'LandContour',
                    'MSZoning', 'Alley', 'LotConfig', 'MiscFeature', 'Condition2',
                    'GarageType', 'RoofStyle', 'SaleType', 'Exterior1st', 'Condition1',
                    'Heating', 'RoofMatl', 'SaleCondition', 'Exterior2nd', 'Foundation',
                    'Street', 'Neighborhood', 'CentralAir']

numerical_features = ['GarageYrBlt', 'GarageArea', 'Area_Qual_Cond_Indicator',
                      'Lot_occupation', 'SalePrice', 'MasVnrArea', 'OverallCond',
                      'BedroomAbvGr', '3SsnPorch', 'MoSold', 'TotalBath', 'BsmtFinSF2',
                      '2ndFlrSF', 'Total_usable_area', 'WoodDeckSF', 'KitchenAbvGr',
                      'HalfBath', 'EnclosedPorch', 'BsmtHalfBath', 'YearRemodAdd',
                      'MiscFeatureExtended', 'BsmtUnfSF', 'House_Age', 'OpenPorchSF',
                      'Lack_of_feature_index', 'YearBuilt', 'HasBsmt', 'Fireplaces',
                      'Total_Close_Live_Area', '1stFlrSF', 'GrLivArea', 'YrSold',
                      'ScreenPorch', 'House_Age2', 'FullBath', 'Has_Alley', 'LotArea',
                      'MSSubClass', 'GarageCars', 'Outside_live_area', 'Has_garage',
                      'TotalBsmtSF', 'Is_Remodeled', 'Area_Quality_Indicator',
                      'BsmtFinSF1', 'OverallQual', 'PoolArea', 'TotRmsAbvGrd',
                      'Quality_conditition_2', 'Number_of_floors', 'LotFrontage',
                      'Quality_conditition', 'LowQualFinSF', 'BsmtFullBath', 'MiscVal']

skewed_features = ['MiscVal', 'PoolArea', 'LotArea', '3SsnPorch', 'LowQualFinSF',
                   'BsmtFinSF2', 'ScreenPorch', 'EnclosedPorch', 'Lot_occupation',
                   'MasVnrArea', 'OpenPorchSF', 'Area_Qual_Cond_Indicator', 'LotFrontage',
                   'WoodDeckSF', 'Area_Quality_Indicator', 'Outside_live_area']

target_column_name = "SalePrice"


GarageQual_map = {'Ex': 5, 'Gd': 4, 'TA': 3,'Fa': 2, 'Po': 1, 'NA': 0}
Fence_map = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0}
GarageFinish_map = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}
KitchenQual_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
GarageCond_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
HeatingQC_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
ExterQual_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
BsmtCond_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
LandSlope_map = {'Gtl': 2, 'Mod': 1, 'Sev': 0}
ExterCond_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
BsmtExposure_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
PavedDrive_map = {'Y': 2, 'P': 1, 'N': 0}
BsmtQual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
LotShape_map = {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0}
BsmtFinType2_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4,
                    'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
BsmtFinType1_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4,
                    'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
FireplaceQu_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
Utilities_map = {"AllPub": 3, "NoSewr": 2, "NoSeWa": 1,  "ELO": 0}
Functional_map = {'Typ': 7, 'Min1': 6, 'Min2': 5,
                  'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}
PoolQC_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}

ordinal_mapping = [{'col': col, 'mapping': globals()[col + '_map']}
                   for col in ordinal_features]

ordinal_transformer = ce.OrdinalEncoder(mapping=ordinal_mapping)

#Created features
Creator = FeatureCreator(add_attributes = True)



@dataclass
class PipelinePathConfig:
    model_obj_file_path=os.path.join('artifacts',"pipeline.pkl")


class PipelineTrainer:
    def __init__(self):
        self.model_config=PipelinePathConfig()

    def configure_pipeline_object(self):
        '''
        This function is responsible for pipeline creation 
        
        '''
        try:

            # TREE PREPROSSESOR
            
            # Preprocessing for numerical data
            numerical_transformer = Pipeline(steps=[

                ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            ])

            # Preprocessing for categorical data
            nominal_transformer = Pipeline(steps=[

                ('imputer', SimpleImputer(strategy='constant', fill_value='Do_not_have_this_feature')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            ordinal_transformer = Pipeline(steps=[    

                ('ordinal_encoder',  ce.OrdinalEncoder(mapping = ordinal_mapping))
            ])      
            
            
            # Bundle preprocessing for tree-based algorithms
            tree_preprocessor = ColumnTransformer(remainder=numerical_transformer,
                transformers=[
                            
                    ('nominal_transformer', nominal_transformer, nominal_features),
                    ('ordinal_transformer', ordinal_transformer, ordinal_features),        
                        
                ])
            
            
            
            # LINEER PREPROSSESOR

            # Preprocessing for numerical data
            numerical_transformer2 = Pipeline(steps=[
                
                ('imputer', SimpleImputer(strategy='constant', fill_value = 0)),
                ('Scaller', StandardScaler()),       
                
            ])

            skewness_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value = 0)),
                ('PowerTransformer', PowerTransformer( method='yeo-johnson', standardize=True)),      
            ])
            
            # Preprocessing for categorical data
            categorical_transformer = Pipeline(steps=[
                
                ('imputer', SimpleImputer(strategy='constant', fill_value = 'Do_not_have_this_feature')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            
            # Bundle preprocessing for linear regression algorithms
            linear_preprocessor = ColumnTransformer(remainder=numerical_transformer2,
                transformers=[
                    ('skewness_transformer', skewness_transformer, skewed_features),
                    ('nominal_transformer', nominal_transformer, nominal_features),
                    ('ordinal_transformer', ordinal_transformer, ordinal_features),

            ])
            
            
            #Algorithm pipelines with found best hyperparameters during development            
            xgb_tunned = XGBRegressor(n_estimators=6500,
                                    alpha=1.7938525031017074e-09,
                                    subsample=0.3231512729662032,
                                    colsample_bytree=0.25528017285233484,
                                    max_depth=5,
                                    min_child_weight=2,
                                    learning_rate=0.004828231865923587,
                                    gamma=0.0026151163125498213,
                                    random_state=1)

            full_pipe_xgb = Pipeline(steps=[
                ('Creator', Creator),
                ('tree_preprocessor', tree_preprocessor),
                ('regressor1', xgb_tunned),
            ])


            gbm_tunned = GradientBoostingRegressor(n_estimators=5500,
                                                max_depth=5,
                                                min_samples_leaf=14,
                                                learning_rate=0.006328507206504974,
                                                subsample=0.9170443266552768,
                                                max_features='sqrt',
                                                random_state=1)


            full_pipe_gbm = Pipeline(steps=[
                ('Creator', Creator),
                ('tree_preprocessor', tree_preprocessor),
                ('regressor2', gbm_tunned),
            ])

            lgbm_tunned = LGBMRegressor(n_estimators=7000,
                                        max_depth=7,
                                        learning_rate=0.002536841439596437,
                                        min_data_in_leaf=22,
                                        subsample=0.7207500503954922,
                                        max_bin=210,
                                        feature_fraction=0.30010067215105635,
                                        random_state=1,
                                        verbosity=-1)

            full_pipe_lgbm = Pipeline(steps=[
                ('Creator', Creator),
                ('tree_preprocessor', tree_preprocessor),
                ('regressor3', lgbm_tunned),
            ])

            catboost_tunned = CatBoostRegressor(iterations=4500,
                                                colsample_bylevel=0.05367479984702603,
                                                learning_rate=0.018477566955501026, random_strength=0.1321272840705348,
                                                depth=6,
                                                l2_leaf_reg=4,
                                                boosting_type='Plain',
                                                bootstrap_type='Bernoulli',
                                                subsample=0.7629052520889268,
                                                logging_level='Silent',
                                                random_state=1)

            full_pipe_catboost = Pipeline(steps=[
                ('Creator', Creator),
                ('tree_preprocessor', tree_preprocessor),
                ('regressor4', catboost_tunned),
            ])


            elasticnet_tunned = ElasticNet(max_iter=3122,
                                        alpha=0.0014964106304254125,
                                        l1_ratio=0.35000000000000003,
                                        tol=3.536319399520495e-06,
                                        random_state=1)

            pipe_Elasticnet = Pipeline(steps=[
                ('Creator', Creator),
                ('linear_preprocessor', linear_preprocessor),
                ('regressor5', elasticnet_tunned),
            ])

            full_pipe_elastic = TransformedTargetRegressor(
                regressor=pipe_Elasticnet, func=np.log1p, inverse_func=np.expm1)


            lasso_tunned = Lasso(max_iter=2345,
                                alpha=0.00019885959230548468,
                                tol=2.955506894549702e-05,
                                random_state=1)

            pipe_Lasso = Pipeline(steps=[
                ('Creator', Creator),
                ('linear_preprocessor', linear_preprocessor),
                ('regressor6', lasso_tunned),
            ])


            full_pipe_lasso = TransformedTargetRegressor(
                regressor=pipe_Lasso, func=np.log1p, inverse_func=np.expm1)


            ridge_tunned = Ridge(max_iter=1537,
                                alpha=6.654338887411367,
                                tol=8.936831872581897e-05,
                                random_state=1)

            pipe_Ridge = Pipeline(steps=[
                ('Creator', Creator),
                ('linear_preprocessor', linear_preprocessor),
                ('regressor7', ridge_tunned),
            ])

            full_pipe_ridge = TransformedTargetRegressor(
                regressor=pipe_Ridge, func=np.log1p, inverse_func=np.expm1)
            
            
            
            #Stacking base estimator pipelines
            estimators = [
                ("full_pipe_xgb", full_pipe_xgb),
                ("full_pipe_gbm", full_pipe_gbm),
                ("full_pipe_lgbm", full_pipe_lgbm),
                ("full_pipe_catboost", full_pipe_catboost),
                ("full_pipe_elastic", full_pipe_elastic),
                ("full_pipe_lasso", full_pipe_lasso),
                ("full_pipe_ridge", full_pipe_ridge),
            ]



            stacking_pipe = StackingRegressor(
                estimators=estimators, final_estimator=LADRegression())


            return stacking_pipe
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def train_pipeline(self,train_path,test_path):
        '''
        This function is responsible for pipeline training 
        
        '''        

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            X_train=train_df.drop(columns=[target_column_name],axis=1)
            y_train=train_df[target_column_name]
            
            X_test=test_df.drop(columns=[target_column_name],axis=1)
            y_test=test_df[target_column_name]
            

            logging.info("Reading train and test data completed")

            pipeline=self.configure_pipeline_object()
            
            logging.info("Pipeline constructed training starting")

            pipeline = pipeline.fit(X_train, y_train)
            preds_test = pipeline.predict(X_test)
            mae = mean_absolute_error(y_test, preds_test)
            
            logging.info("Training finished, mean absolute error on testing is {mae}".format(mae = mae))

 
            save_object(

                file_path=self.model_config.model_obj_file_path,
                obj=pipeline

            )
            
            logging.info(f"Trained pipeline object is saved.")

            return mae
        
        except Exception as e:
            raise CustomException(e,sys)