# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 00:36:41 2020

@author: Balaji
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.api as sm
import xgboost as xgb
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

df = pd.read_csv('C:/Users/1636740/Desktop/tshr/backup/bakcup/datasets_14872_228180_Admission_Predict_Ver1.1/datasets_14872_228180_Admission_Predict_Ver1.1.csv')
# print(df.describe())
# print(df.info())
# print(df.columns)
df = df.drop(['Serial No.'], axis= 1)
# print(df.isnull().sum())

plt.figure(figsize=[12,10])
sns.pairplot(df,kind="reg",hue="Research", palette="husl")

plt.figure(figsize = (12,10))
sns.heatmap(df.corr(),cmap='BrBG', annot = True, linecolor = 'white', linewidth = 1)
plt.tight_layout()

for i in df.drop(['Chance of Admit ','Research','CGPA','GRE Score','TOEFL Score'],axis =1).columns:
    plt.figure(figsize = (12,8))
    ax = sns.boxplot(x = i,y = 'Chance of Admit ',data = df,palette = 'Set3', hue = 'Research')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.show()



for i in df.drop(['Chance of Admit ','Research','CGPA','GRE Score','TOEFL Score'],axis =1).columns:
    plt.figure(figsize = (12,8))
    ax = sns.boxplot(x = i,y = 'Chance of Admit ',data = df,palette = 'Set3', hue = 'Research')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.show()


for i in df.drop(['Chance of Admit ','Research','CGPA','GRE Score','TOEFL Score'],axis =1).columns:
    plt.figure(figsize = (12,8))
    ax = sns.barplot(x = i,y = 'Chance of Admit ',data = df,palette="viridis",hue = 'Research')
    plt.tight_layout()
    plt.xticks( rotation=90)
    plt.show()
    
for i in df.drop(['Chance of Admit '],axis =1).columns:
    sns.jointplot(df[i], 'Chance of Admit ',kind="hex", color='#4CB391', data = df, space = 0.1) 
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    sns.jointplot(df[i], 'Chance of Admit ', data=df, kind="kde", color = 'purple',n_levels=10, cmap=cmap, shade = True)


for i in df[['TOEFL Score','GRE Score','CGPA']]:
    fig = plt.figure(figsize=(8, 5))
    sns.lmplot(i, 'Chance of Admit ',data = df,hue ='Research',palette = 'husl')
    plt.grid()
    plt.show()

for i in df.drop(['Research'],axis =1).columns:
    plt.figure(figsize =(6,4)) 
    ax = sns.distplot(df[i], hist=True, color="red", kde = True,kde_kws={"shade": True}, hist_kws = dict(edgecolor="k", linewidth=3))
    plt.tight_layout()

for i in df.drop(['Research','Chance of Admit ','CGPA'],axis =1).columns:
    plt.figure(figsize=[10,8])
    sns.scatterplot(x=df[i], y=df['Chance of Admit '], hue = df['Research'],size=df['CGPA'],sizes=(10, 200),palette ='terrain')

np.random.seed(100)
y = df['Chance of Admit ']
X = df.drop(['Chance of Admit '], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc_X = StandardScaler()
# df = sc_X.fit_transform(df)
X2_train = sc_X.fit_transform(X_train)
X2_test = sc_X.transform(X_test)

# adsaasd
# X_sm = sm.add_constant(X)
# model = sm.OLS(y,X_sm)
# print(model.fit().summary())

lm1 = LinearRegression()
lm1.fit(X_train,y_train)
lm1_pred = lm1.predict(X_test)
print('Linear Regression Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, lm1_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lm1_pred)))
print('R2_Score: ', metrics.r2_score(y_test, lm1_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,lm1_pred,color='g')
plt.xlabel('COA') 
plt.ylabel('Predictions') 
plt.title('LinearRegression Prediction Performance ')  
plt.grid()
plt.show()
# print('Estimated coefficients for the linear regression: ',lm1.coef_)
# print('Independent term: ', lm1.intercept_)

# import pickle
# filename = 'LinearRegression.sav'
# pickle.dump(lm1, open(filename, 'wb'))
# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)

# ls = Lasso()
# ls.fit(X_train,y_train)
# ls_pred = ls.predict(X_test)
# print('Lasso Regression Performance:')
# print('MAE:', metrics.mean_absolute_error(y_test, ls_pred))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, ls_pred)))
# print('R2_Score: ', metrics.r2_score(y_test, ls_pred))
# fig = plt.figure(figsize=(8, 5))
# sns.regplot(y_test,ls_pred,color='g')
# plt.xlabel('COA') 
# plt.ylabel('Predictions') 
# plt.title('Lasso Prediction Performance ') 
# plt.grid()

rg = Ridge()
rg.fit(X_train,y_train)
rg_pred = rg.predict(X_test)
print('Ridge Regression Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, rg_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rg_pred)))
print('R2_Score: ', metrics.r2_score(y_test, rg_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,rg_pred,color='g')
plt.xlabel('COA') 
plt.ylabel('Predictions') 
plt.title('Ridge Prediction Performance ') 
plt.grid()

# rf = RandomForestRegressor(n_estimators=100)
# rf.fit(X_train,y_train)
# rf_pred = rf.predict(X_test)
# print('RandomForestRegressor Performance:')
# print('MAE:', metrics.mean_absolute_error(y_test, rf_pred))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rf_pred)))
# print('R2_Score: ', metrics.r2_score(y_test, rf_pred))
# fig = plt.figure(figsize=(8, 5))
# sns.regplot(y_test,rf_pred,color='g')
# plt.xlabel('COA') 
# plt.ylabel('Predictions') 
# plt.title('RFR Prediction Performance ') 
# plt.grid()
# plt.show()
# feat_imp = pd.Series(rf.feature_importances_, list(X_train)).sort_values(ascending=False)
# fig = plt.figure(figsize=(12, 6))
# feat_imp.plot(kind='bar', title='Importance of Features',color = 'coral')
# plt.ylabel('Feature Importance Score')
# plt.grid()
# plt.show()


# rf_param_grid = {'max_features': ['sqrt', 'auto','log2'],
#               'min_samples_leaf': [1, 3, 5],
#               'n_estimators': [100, 500, 1000],
#               'bootstrap': [False, True],
#               'criterion' : ['mse','mae']}

# rf_grid = GridSearchCV(estimator= RandomForestRegressor(), param_grid = rf_param_grid,  n_jobs=-1, verbose=0)
# rf_grid.fit(X_train,y_train)
# print(rf_grid.best_params_)
# print(rf_grid.best_estimator_)

rf_T = RandomForestRegressor(bootstrap=True, criterion='mae',
                      max_features='log2', min_samples_leaf=1,
                      min_samples_split=2, n_estimators=100,
                      verbose=0)
rf_T.fit(X_train,y_train)
rf_T_pred = rf_T.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, rf_T_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rf_T_pred)))
print('R2_Score: ', metrics.r2_score(y_test, rf_T_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,rf_T_pred,color='g')
plt.xlabel('COA') 
plt.ylabel('Predictions') 
plt.title('RFRT Prediction Performance ')  
plt.grid()
plt.show()
feat_imp = pd.Series(rf_T.feature_importances_, list(X_train)).sort_values(ascending=False)
fig = plt.figure(figsize=(12, 6))
feat_imp.plot(kind='bar', title='Importance of Features',color = 'coral')
plt.ylabel('Feature Importance Score')
plt.grid()
plt.show()


# gbm1 = GradientBoostingRegressor(n_estimators=100)
# gbm1.fit(X_train, y_train)
# gbm1_pred = gbm1.predict(X_test)
# print('Gradiant Boosting Performance:')
# print('MAE:', metrics.mean_absolute_error(y_test, gbm1_pred))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, gbm1_pred)))
# print('R2_Score: ', metrics.r2_score(y_test, gbm1_pred))
# fig = plt.figure(figsize=(8, 5))
# sns.regplot(y_test,gbm1_pred,color='g')
# plt.xlabel('COA') 
# plt.ylabel('Predictions') 
# plt.title('GBM Prediction Performance ') 
# plt.grid()
# plt.show()
# feat_imp = pd.Series(gbm1.feature_importances_, list(X_train)).sort_values(ascending=False)
# fig = plt.figure(figsize=(8, 5))
# feat_imp.plot(kind='bar', title='Importance of Features', color ='coral')
# plt.ylabel('Feature Importance Score')
# plt.grid()
# plt.show()

# gbm_param_grid = {'learning_rate':[1,0.1, 0.01, 0.001], 
#             'n_estimators':[100, 500, 1000],
#           'max_depth':[3, 5, 8],
#           'subsample':[0.7, 1], 
#           'min_samples_leaf':[1, 20],
#           'min_samples_split':[10, 20],
#           'max_features':[4, 7]}
# gbm_tuning = GridSearchCV(estimator =GradientBoostingRegressor(random_state=101),
#                           param_grid = gbm_param_grid,
#                           n_jobs=-1,
#                           cv=5)
# gbm_tuning.fit(X_train,y_train)
# print(gbm_tuning.best_params_)
# print(gbm_tuning.best_estimator_)

gbmT = GradientBoostingRegressor(learning_rate=0.01,  max_depth=3,
                          max_features=4,
                          min_samples_leaf=1, min_samples_split=20,
                          n_estimators=500)
gbmT.fit(X_train, y_train)
gbmT_pred = gbmT.predict(X_test)
print('Gradiant Boosting Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, gbmT_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, gbmT_pred)))
print('R2_Score: ', metrics.r2_score(y_test, gbmT_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,gbmT_pred,color='g')
plt.xlabel('COA') 
plt.ylabel('Predictions') 
plt.title('GBMT Prediction Performance ')  
plt.grid()
plt.show()
feat_imp = pd.Series(gbmT.feature_importances_, list(X_train)).sort_values(ascending=False)
fig = plt.figure(figsize=(8, 5))
feat_imp.plot(kind='bar', title='Importance of Features', color ='coral')
plt.ylabel('Feature Importance Score')
plt.grid()
plt.show()



# svr = SVR()
# svr.fit(X_train,y_train)
# svr_pred = svr.predict(X_test)
# print('SVR Performance:')
# print('MAE:', metrics.mean_absolute_error(y_test, svr_pred))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))
# print('R2_Score: ', metrics.r2_score(y_test, svr_pred))
# fig = plt.figure(figsize=(8, 5))
# sns.regplot(y_test,svr_pred,color='g')
# plt.xlabel('COA') 
# plt.ylabel('Predictions') 
# plt.title('SVR Prediction Performance ') 
# plt.grid()
# plt.show()


# # # param_grid = {'C': [1, 10, 100], 'gamma': [0.01,0.001,0.0001], 'kernel': ['rbf']} 
# # # grid = GridSearchCV(SVR(),param_grid,refit=True,verbose=3)
# # # grid.fit(X_train,y_train)
# # # print(grid.best_params_)
# # # print(grid.best_estimator_)

# svrT = SVR(C=100,gamma=0.001,kernel='rbf')
# svrT.fit(X_train,y_train)
# svr_pred = svrT.predict(X_test)
# print('SVRT Performance:')
# print('MAE:', metrics.mean_absolute_error(y_test, svr_pred))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))
# print('R2_Score: ', metrics.r2_score(y_test, svr_pred))
# fig = plt.figure(figsize=(8, 5))
# sns.regplot(y_test,svr_pred,color='g')
# plt.xlabel('COA') 
# plt.ylabel('Predictions') 
# plt.title('SVRT Prediction Performance ') 
# plt.grid()
# plt.show()

# xbgr = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 10)
# xbgr.fit(X_train,y_train)
# xbgr_pred = xbgr.predict(X_test)
# print('XGB Performance:')
# print('MAE:', metrics.mean_absolute_error(y_test, xbgr_pred))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xbgr_pred)))
# print('R2_Score: ', metrics.r2_score(y_test, xbgr_pred))
# fig = plt.figure(figsize=(8, 5))
# sns.regplot(y_test,xbgr_pred,color='g')
# plt.xlabel('COA') 
# plt.ylabel('Predictions') 
# plt.title('GBMT Prediction Performance ')  
# plt.grid()
# plt.show()
# feat_imp = pd.Series(xbgr.feature_importances_, list(X_train)).sort_values(ascending=False)
# fig = plt.figure(figsize=(8, 5))
# feat_imp.plot(kind='bar', title='Importance of Features', color ='coral')
# plt.ylabel('Feature Importance Score')
# plt.grid()
# plt.show()

# params = {"learning_rate"    : [0.05,  0.15, 0.25 ],
#   "max_depth"        : [ 3,  5,  8,  12 ],
#   "min_child_weight" : [ 1, 3, 5, 7 ],
#   "gamma"            : [ 0.0, 0.1, 0.2 , 0.3,],
#   "colsample_bytree" : [ 0.3, 0.4, 0.5 ],
#   'n_estimators': [ 100, 300, 500]}
# xgb_tuning = GridSearchCV(estimator = xgb.XGBRegressor(random_state=101),
#                           param_grid = params,
#                           n_jobs=-1,
#                           cv=3)
# xgb_tuning.fit(X_train,y_train)
# print(xgb_tuning.best_params_)
# print(xgb_tuning.best_estimator_)



xbgrT = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.05, max_delta_step=0, max_depth=3,
              min_child_weight=7, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='reg:squarederror', random_state=101, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
              validate_parameters=1, verbosity=None)
xbgrT.fit(X_train,y_train)
xbgrT_pred = xbgrT.predict(X_test)
print('XGBT Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, xbgrT_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xbgrT_pred)))
print('R2_Score: ', metrics.r2_score(y_test, xbgrT_pred))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,xbgrT_pred,color='g')
plt.xlabel('COA') 
plt.ylabel('Predictions') 
plt.title('GBMT Prediction Performance ')  
plt.grid()
plt.show()
feat_imp = pd.Series(xbgrT.feature_importances_, list(X_train)).sort_values(ascending=False)
fig = plt.figure(figsize=(8, 5))
feat_imp.plot(kind='bar', title='Importance of Features', color ='coral')
plt.ylabel('Feature Importance Score')
plt.grid()
plt.show()



# adasdasdsa

import tensorflow as tf 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# sc_X = StandardScaler()
# X2_train = sc_X.fit_transform(X_train)
# X2_test = sc_X.transform(X_test)
# test2 = sc_X.transform(test)


model = Sequential()

model.add(Dense(7, activation = 'relu'))
# model.add(Dropout(0.3))

model.add(Dense(5, activation = 'relu'))
# model.add(Dropout(0.2))

model.add(Dense(2, activation = 'relu'))
# model.add(Dropout(0.2))

# model.add(Dense(3, activation = 'relu'))
# model.add(Dropout(0.2))

model.add(Dense(1, activation = 'relu'))

model.compile(loss = 'mse', optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss' , mode = 'min', verbose=1, patience=10)
# =============================================================================
# Kfold in DL
# =============================================================================
# def create_baseline():
#     # create model
#     model = Sequential()

#     model.add(Dense(7, activation = 'relu'))
#     # model.add(Dropout(0.3))
    
#     model.add(Dense(5, activation = 'relu'))
#     # model.add(Dropout(0.2))
    
#     model.add(Dense(2, activation = 'relu'))
#     # model.add(Dropout(0.2))
    
#     # model.add(Dense(3, activation = 'relu'))
#     # model.add(Dropout(0.2))
    
#     model.add(Dense(1, activation = 'relu'))
    
#     model.compile(loss = 'mse', optimizer='adam')
    
#     # early_stop = EarlyStopping(monitor='val_loss' , mode = 'min', verbose=1, patience=10)

#     return model
# # evaluate model with standardized dataset
# estimator = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_baseline, epochs=10, batch_size=5, verbose=0 )
# kfold = KFold(n_splits=10, shuffle=True, random_state=42)
# results = cross_val_score(estimator, X2_train, y_train, cv=kfold, scoring = 'neg_mean_squared_error')
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# =============================================================================
# 
# =============================================================================

model.fit(x=X2_train, y = y_train, epochs=1000, validation_data=(X2_test,y_test), callbacks=[early_stop])


losses = pd.DataFrame(model.history.history)

fig = plt.figure(figsize=(10,8))
losses.plot()

predictions = model.predict(X2_test)

print('NN Performance:')
print('MAE:', metrics.mean_absolute_error(np.array(y_test), predictions))
print('MSE:', metrics.mean_squared_error(np.array(y_test), predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(np.array(y_test), predictions)))
print('R2_Score: ', metrics.r2_score(np.array(y_test), predictions))
fig = plt.figure(figsize=(8, 5))
sns.regplot(np.array(y_test), predictions,color='g')
plt.xlabel('COA') 
plt.ylabel('Predictions') 
plt.title('NN Prediction Performance ')  
plt.grid()
plt.show()




# #Evaluation
# Models = ['Linear Regression','Lasso Regression','RandomForestRegressorT','Gradiant BoostingT','SVR','XGBT','Ridge']
# MAEs = [0.04085966289143846,0.09191308963175515,0.044427666666666664, 0.04552636749116818,0.06260443779506134,0.04442984240849813,0.045187706811545054]
# Scores = [0.8260541587118344, 0.283059678676331,0.8027016231774982,0.8232384797586916,0.7091498280257977,0.8267744263896804,0.8368737677028406]

# fig = plt.figure(figsize=(8, 5))
# plt.plot(Models, MAEs, 'o', color='red',
#           markersize=10, linewidth=3,
#           markerfacecolor='gray',
#           markeredgecolor='k',
#           markeredgewidth=1, ls='--')
# plt.xlabel('Model') 
# plt.xticks( rotation=90)
# plt.ylabel('MAE') 
# plt.title('Model vs MAE') 
# plt.grid()
# plt.show()

# fig = plt.figure(figsize=(8, 5))
# plt.plot(Models, Scores, 'o', color='red',
#           markersize=10, linewidth=3,
#           markerfacecolor='gray',
#           markeredgecolor='k',
#           markeredgewidth=1, ls='--')
# plt.xlabel('Model')
# plt.xticks( rotation=90)
# plt.ylabel('R2_Score') 
# plt.title('Model vs R2_Score') 
# plt.grid()
# plt.show()

print('Final Graph of actual vs Pred: ')
pred = (lm1.predict(X_test) + gbmT.predict(X_test) + rf_T.predict(X_test) + xbgrT.predict(X_test) + rg.predict(X_test))/5# + predictions)/6
preds = (pred + predictions)/2
preds = np.diag(preds)
y_test = np.array(y_test)
print('Final Performance:')
print('MAE:', metrics.mean_absolute_error(y_test, preds))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds)))
print('R2_Score: ', metrics.r2_score(y_test, preds))
fig = plt.figure(figsize=(8, 5))
sns.regplot(y_test,preds,color='g')
plt.xlabel('y_test_actual') 
plt.ylabel('y_test_actual_predicted') 
plt.title('actual vs Pred')  
plt.grid()
plt.show()

print('Average Mean_Absolute_Error is {}'.format(round(metrics.mean_absolute_error(y_test, preds),4)))
print('Average Mean_Squared_Error is {}'.format(round(metrics.mean_squared_error(y_test, preds),4)))
print('Average Root Mean_Squared_Error is {}'.format(round(np.sqrt(metrics.mean_squared_error(y_test, preds)),4)))
print('Average prediction accuracy is {}% '.format(round(metrics.r2_score(y_test, preds)*100,2)))


# # CASE STUDY:

import random

GRE = []
TOEFL = []
UR = []
SOP = []
LOR = []
CGPA = []
Research = []


for i in range(0,25):
    GRE.append(random.choice(range(295,341,1)))
    TOEFL.append(random.choice(range(90,121,1)))
    UR.append(random.choice(range(1,6,1)))
    SOP.append(random.choice(range(10,55,5))/10)
    LOR.append(random.choice(range(10,55,5))/10)
    CGPA.append(round(random.uniform(7.5,10.1),2))
    Research.append(random.choice(range(0,2,1)))
    

df_cs = pd.DataFrame(list(zip(GRE, TOEFL, UR, SOP ,LOR, CGPA, Research)), 
                columns =['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']) 


lm_cs_pred = lm1.predict(df_cs)
gbm_cs_pred = gbmT.predict(df_cs)
rf_cs_pred = rf_T.predict(df_cs)
xbgrT_cs_pred = xbgrT.predict(df_cs)
rg_cs_pred = rg.predict(df_cs)
NN_cs_pred = model.predict(sc_X.transform(df_cs))

COA_cs6 = (lm_cs_pred + gbm_cs_pred + rf_cs_pred + xbgrT_cs_pred + rg_cs_pred + NN_cs_pred)/6
COA_cs6 = pd.Series(np.diag(COA_cs6))
results_comp = pd.DataFrame(list(zip(lm_cs_pred,rg_cs_pred,gbm_cs_pred,rf_cs_pred,xbgrT_cs_pred,NN_cs_pred,COA_cs6)), 
                columns =['lm_cs_pred', 'rg_cs_pred','gbm_cs_pred','rf_cs_pred','xbgrT_cs_pred','NN_cs_pred','COA_cs_ave']) 

df_cs[['lm_cs_pred','rg_cs_pred', 'gbm_cs_pred','rf_cs_pred','xbgrT_cs_pred','NN_cs_pred','COA_cs_ave']] = results_comp[['lm_cs_pred', 'rg_cs_pred', 'gbm_cs_pred','rf_cs_pred','xbgrT_cs_pred','NN_cs_pred','COA_cs_ave']]

df_cs.to_csv('C:/Users/Balaji/Documents/tshr/python_projects/datasets_14872_228180_Admission_Predict_Ver1.1/CaseStudy.csv')

# #garbage
# # final1, final2, final3 ,final4, final5, final6, final7 = [],[],[],[],[],[],[]
# # for i in df_cs_results:
# #     if abs(df_cs_results[lm_cs_pred[i]] - df_cs_results[gbm_cs_pred[i]]) > 0.45:
# #         final1[i] = final1.append(max(lm_cs_pred[i],gbm_cs_pred[i]))
# #     elif abs(lm_cs_pred[i] - rf_cs_pred[i]) > 0.45:
# #         final2[i] = final2.append(max(lm_cs_pred[i],rf_cs_pred[i]))
# #     elif abs(gbm_cs_pred[i] - rf_cs_pred[i]) > 0.45:
# #         final3[i] = final3.append(max(gbm_cs_pred[i],rf_cs_pred[i]))
# #     elif abs(gbm_cs_pred[i] - lm_cs_pred[i]) > 0.45:
# #         final4[i] = final4.append(max(gbm_cs_pred[i],lm_cs_pred[i]))
# #     elif abs(rf_cs_pred[i] - lm_cs_pred[i]) > 0.45:
# #         final5[i] = final5.append(max(rf_cs_pred[i],lm_cs_pred[i]))
# #     elif abs(rf_cs_pred[i] - gbm_cs_pred[i]) > 0.45:
# #         final6[i] = final6.append(max(rf_cs_pred[i],gbm_cs_pred[i]))
# #     else :
# #         final7[i] = final7.append((lm_cs_pred+gbm_cs_pred+rf_cs_pred)/3)

# # final_ = (final1+final2+final3+final4+final5+final6)/6

