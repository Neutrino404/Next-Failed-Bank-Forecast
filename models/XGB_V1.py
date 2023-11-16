#In this notebook, I attempt to predict the next bank failure using historic financial ratios for all U.S. banks in the FDIC database. 
#The target labels are binary, with a positive instance indicating that the bank will fail within a given time frame.
#I have specifically chosen the failure within 180 days to optimize for the fact that the FDIC releases the quarterly bank call report data with a lag of about 60 to 90 days after the end of the quarter. 
#Also, this label seemed to give the best results. I also prioritized precision over recall, as shown in the confusion matrix below for the best model. 
#The hyperparameters were selected via HyperOpt in another private notebook and are used below. 
#Also, the results for different sets of hyperparameters are also recorded below. 
#Financial ratios dataset is of shape (656438, 796) with 796 features and 656438 training examples for the past 90 quarters (see dataset page for more information).
#The best model achieved the following results on the test set: AUC=0.636, F1=0.413, Precision=0.83

#The code below was run in a google colab notebook with gpu enabled and a High-Ram instance 

#Imports
import numpy as np
import pandas as pd
import xgboost as xg
import os
from google.colab import runtime
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
import zipfile
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

!kaggle
#Get access to kaggle data via api key, first upload kaggle.json then move it to the correct location with the code below
!mv kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json


#Downloading files
!kaggle datasets download -d neutrino404/failed-banks-fdic-data/
!kaggle datasets download -d neutrino404/all-us-banks-financial-quarterly-ratios-2001-2023
with zipfile.ZipFile('failed-banks-fdic-data.zip', 'r') as zip_ref:
    zip_ref.extractall('failed_banks')
with zipfile.ZipFile('all-us-banks-financial-quarterly-ratios-2001-2023.zip', 'r') as zip_ref:
    zip_ref.extractall('ratios')
  
  
#Creating dataframes
failed_df = pd.read_csv('failed_banks/Failed_Bank_Dataset_2.csv')
ratios_df = pd.read_csv('ratios/total_bank_data.csv')


#Merging dataframes
merged_df = pd.merge(ratios_df, failed_df[['Closing_Date','Cert']], left_on='CERT', right_on='Cert', how='left')


#Clearing Memory 
del ratios_df
del failed_df


#Sorting dataframe
merged_df.sort_values(by='REPDTE', inplace=True)
merged_df = merged_df.reset_index(drop=True)


#Converting dates to correct format
merged_df['REPDTE'] = pd.to_datetime(merged_df['REPDTE'], format='%Y%m%d')
merged_df['Closing_Date'] = pd.to_datetime(merged_df['Closing_Date'])


#Making target label column
merged_df['FailWithin180Days'] = ((merged_df['Closing_Date'] - merged_df['REPDTE']).dt.days >= 0) & ((merged_df['Closing_Date'] - merged_df['REPDTE']).dt.days <= 180)


#Finding split locations for a 70/15/15 split
lower_date = pd.to_datetime(merged_df['REPDTE'][int(merged_df.shape[0]*0.70)])
top_date = pd.to_datetime(merged_df['REPDTE'][int(merged_df.shape[0]*0.85)])
last_date = pd.to_datetime(merged_df['REPDTE'].iloc[-1])


#Dropping unnecessary features 
merged_df.drop(['CBLRINDQ','Closing_Date','REPYEAR','STNAME','Cert'], axis=1, inplace=True)


#Splitting dataframe into test (0.7)/val (0.15)/test (~0.15) sets and last date
train_data = merged_df[merged_df['REPDTE'] < lower_date]
validation_data = merged_df[(merged_df['REPDTE'] >= lower_date) & (merged_df['REPDTE'] < top_date)]
test_data = merged_df[(merged_df['REPDTE'] >= top_date) & (merged_df['REPDTE'] < last_date)]
last_data = merged_df[merged_df['REPDTE'] >= last_date]

#Dropping unnecessary features 
train_data.drop(['REPDTE','NAME','CERT'], axis=1, inplace=True)
validation_data.drop(['REPDTE','NAME','CERT'], axis=1, inplace=True)
test_data.drop(['REPDTE','NAME','CERT'], axis=1, inplace=True)


#Resetting index
Y_train = Y_train.reset_index(drop=True)
Y_validation = Y_validation.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)
last_data = last_data.reset_index(drop=True)


#Making a dataframe with NAME CERT columns that correspond to the last date
last_data_ids = last_data[['NAME','CERT']].copy()
last_data.drop(['REPDTE','NAME','CERT'], axis=1, inplace=True)


Clearing Memory 
del merged_df

#Splitting label column 
Y2t = train_data['FailWithin180Days']
Y2v = validation_data['FailWithin180Days']
Y2s = test_data['FailWithin180Days']

Y_train = Y2t
train_data.drop(['FailWithin180Days'], axis=1, inplace=True)
Y_val = Y2v
validation_data.drop(['FailWithin180Days'], axis=1, inplace=True)
Y_test = Y2s
test_data.drop(['FailWithin180Days'], axis=1, inplace=True)

last_data.drop(['FailWithin180Days'], axis=1, inplace=True)





#Training with the best set of hyperparameters 
params = {'objective': 'binary:logistic', 'base_score': None, 'booster': None, 'callbacks': None, 'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': 1.0, 'device': 'cuda', 'early_stopping_rounds': None, 'enable_categorical': False, 'eval_metric': None, 'feature_types': None, 'gamma': 0.8, 'grow_policy': None, 'importance_type': None, 'interaction_constraints': None, 'learning_rate': None, 'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': 8, 'max_leaves': None, 'min_child_weight': None,'monotone_constraints': None, 'multi_strategy': None, 'n_estimators': 240, 'n_jobs': None, 'num_parallel_tree': None, 'random_state': None, 'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': None, 'subsample': 0.8, 'tree_method': 'gpu_hist', 'validate_parameters': None, 'verbosity': None, 'eta': 0.30000000000000004, 'seed': 36}

xgb = xg.XGBClassifier(**params)

eval_set = [(train_data, Y_train), (validation_data, Y_val)]
xgb.fit(train_data, Y_train,
# Fitting the model
          eval_set = eval_set,
          verbose=100)




#Viewing metrics for results 
def show_results(X_data,Y_data):
  y_pred_binary = model.predict(X_data)
  
  f1_scores = f1_score(Y_data, y_pred_binary)
  aucpr = average_precision_score(Y_data, y_pred_binary)
  precision = precision_score(Y_data, y_pred_binary)
  
  print(f'Precision: {precision}')
  print(f'AUCPR: {aucpr}')
  print(f'F1 Score: {f1_scores}')

show_results(validation_data,Y_validation)
print('##############################')
show_results(test_data,Y_test)




# Create a confusion matrix
def C_matrix(X_data,Y_data):
  y_pred_binary = model.predict(X_data)
  cm = confusion_matrix(Y_data, y_pred_binary)
  # Display the confusion matrix as a heatmap
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"])
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title("Confusion Matrix")
  plt.show()
  print("Confusion Matrix:")
  print(cm)

C_matrix(test_data,Y_test)


#Confustion matrix for test data using top parameters: 
#[[96805     1]
#[   17     5]]
#The model has a F1=0.413, Precision=0.83 

#Predictiong potential failing banks within 180 days since last quarter
y_pred= xgb.predict_proba(last_data)
last_data_ids['Pred'] = y_pred
sorted_results = last_data_ids.sort_values(by="Pred",ascending=False)
print(sorted_results[:20])






























