# Admission_Predictor
### A ML Regression project to predict chance of an admit

This projects helps predicting the chance of an admit (in %) provided the following features:
1. GRE Scores: (Out of 340)
2. TOEFL Scores: (Out of 120)
3. University Rating: (Out of 5)
4. SOP Rating (Statement of Purpose): (Out of 5)
5. LOR Rating (Letter of Recommendation): (Out of 5)
6. CGPA: (Out of 10)
7. Research Opportunity:  (1 or 0)


## Project Structure

Admission_Predict.ipynb file gives the walkthrough over the complete project. 
Model.pkl file is the saved pickle file of final model.
CaseStudy.csv is the csv file of Case Study considered for prediction

app.py file gives the walkthrough over the deployment of project in flask. 
LinearRegression.sav is the saved file of model used in flask.

plots folder contains the saved images of all plots of EDA and model performances.

templates folder contains the html template.


## To run the prject, follow below steps
1.  Ensure that you are in the project home directory
2.  Create anaconda environment
3.  Activate environment
4.  >pip install -r requirement.txt
5.  >python app.py
6.  Navigate to URL http://localhost:5000


### Please feel free to connect for any suggestions or doubts!!!


## Credits
1.  I have modified https://github.com/pik1989/MLProject-ChurnPrediction/tree/main/templates html template for flask
2.  The credit for dataset used for training goes to https://www.kaggle.com/mohansacharya/graduate-admissions
3.  The credit for image used in html file for background goes to https://blog.unpakt.com/wp-content/uploads/2017/06/63064200_l-education-graduation-and-people-concept-silhouettes-of-many-happy-students-in-gowns-throwing-mortarboards-in-air-1024x651.jpg
