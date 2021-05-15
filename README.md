# Admission_Predictor
### A ML Regression project to predict chance of an admit

This projects helps predicting the chance of an admit (in %) provided the following features:
  GRE Scores: (Out of 340)
  TOEFL Scores: (Out of 120)
  University Rating: (Out of 5)
  SOP Rating (Statement of Purpose): (Out of 5)
  LOR Rating (Letter of Recommendation): (Out of 5)
  CGPA: (Out of 10)
  Research Opportunity:  (1 or 0)


The credit for dataset used for training goes to https://www.kaggle.com/mohansacharya/graduate-admissions


Admission_Predict.ipynb file gives the walkthrough over the complete project. 
Model.pkl file is the saved pickle file of final model.
CaseStudy.csv is the csv file of Case Study considered for prediction

app.py file gives the walkthrough over the deployment of project in flask. 
LinearRegression.sav is the saved file of model used in flask.

plots folder contains the saved images of all plots of EDA and model performances


To run the prject, follow below steps:
1.  Create anaconda environment
2.  Activate environment
3.  >pip install -r requirement.txt
4.  >python app.py


Please feel free to connect for any suggestions or doubts!!!
