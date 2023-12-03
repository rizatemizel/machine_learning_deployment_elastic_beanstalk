
# End-to-End Housing Price Prediction Project
This repo contains an end-to-end data science project on Kaggle's housing price data set.

My work was the best-performing shared notebook among over 55 thousand trials in Kaggle's machine-learning competition for housing price prediction. You can find the original notebook at this link: https://www.kaggle.com/code/rzatemizel/feature-engineering-optuna-stacked-pipe

Here, I used the same notebook with some little refinements for deployment. 

I wanted this project to be quickly adaptable for any tabular data set. I tried to keep it modular and retrainable.

Finally, I deployed the project on AWS Elastic Beanstalk as a FastAPI based web application.
## Brief Overview of File Structure
.ebextensions: Configuration for elastic beanstalk deployment
artifacts: contains training data and trained pipeline as pkl
model_development: includes model development notebook
src: contains data ingestion, feature engineering, and training modules with some utilities
-Procfile and appplication.py: These are specifically for FastAPI deployment
## Tech Stack

**For model development:** 
- Python, Pandas, Scikit-learn, Seaborn, Optuna



**For deployment:** 
- FastAPI, AWS Elastic Beanstalk


## Model Development Approach
You can check model_development_notebook or Kaggle link for development details. These are almost annoyingly instructive written notebooks.

Very briefly:
An ensemble model with various base models is used for training

Scikit-learn pipelines are heavily used for preprocessing, feature engineering, training, and ensembling. Pipelines made it possible to deploy my project quickly with a single pkl file. 

Optuna package is used for Bayesian Hyperparameter optimization.


![pipeline](https://github.com/rizatemizel/ml_deployment/assets/127015640/e80b8e2f-eb3e-400c-b152-75f1c7f3a085)


## Deployment
I deployed the project on the AWS Elastic Beanstalk environment as a FastAPI.

The .ebextensions folder, procfile, and appplication.py files are required configurations for this purpose.
In addition, you need to create an application and associated environment in Elastic Beanstalk and deploy your application via AWS code pipeline from a GitHub repository.

What does it look like at the end?

![elastic_beanstalk](https://github.com/rizatemizel/ml_deployment/assets/127015640/c6c22c6a-733d-4306-a3aa-5fedaadea2c4)

How does it serve predictions?
![elastic_beanstalk2](https://github.com/rizatemizel/ml_deployment/assets/127015640/899dca35-3cd5-4dcd-9070-a68726d6edba)




## Lessons Learned
I encountered two issues while deploying Elastic Beanstalk:

1- records in eb-engine log say: ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device

None of the suggested ways on the internet worked for me to solve this issue. I used a different instance type with more memory to solve the problem. (To increase EBS volume on free tier instance didn't work in my case.)

2- Procfile is important. Remember to include it on FastAPI-based applications.
