# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#library used
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


#Changing the working directory
os.getcwd()
os.chdir("D:")


name=[]
for i in range(785):
    name.append(i)
    
#Loading the data and looking into the data type
df = pd.read_csv("fashion_train.csv" , names = name)
df_test = pd.read_csv("fashion_test.csv", names = name)

#df.columns
#
#def LogisticR(data):
#    for i in range(len(data)):
#        if data.iloc[i,0] == 0:
#            data.iloc[i,0] = 1
#        else:
#            data.iloc[i,0] = 0 
#    print(data.head(5))
#    
#LogisticR(df)


# logistic regresion for class 0
df0 = df.copy(deep=True)

df0_test = df_test.copy(deep=True)

dfs = {}
for i in range(10):
    dfs[i] = df.copy(deep=True)

dfs_test = {}
for i in range(10):
    dfs_test[i] = df_test.copy(deep=True)
#for i in range(len(df0)):
#    if df0.iloc[i,0] == 0:
#        df0.iloc[i,0] = 1
#    else:
#        df0.iloc[i,0] = 0 

for i in range(10):
    if i == 0:
        dfs[0][0] = dfs[0][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [1,0,0,0,0,0,0,0,0,0])
    if i == 1:
        dfs[1][0] = dfs[1][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,1,0,0,0,0,0,0,0,0])
    if i == 2:
        dfs[2][0] = dfs[2][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,1,0,0,0,0,0,0,0])
    if i == 3:
        dfs[3][0] = dfs[3][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,0,1,0,0,0,0,0,0])
    if i == 4:
        dfs[4][0] = dfs[4][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,0,0,1,0,0,0,0,0])     
    if i == 5:
        dfs[5][0] = dfs[5][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,0,0,0,1,0,0,0,0])
    if i == 6:
        dfs[6][0] = dfs[6][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,0,0,0,0,1,0,0,0])
    if i == 7:
        dfs[7][0] = dfs[7][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,0,0,0,0,0,1,0,0])
    if i == 8:
        dfs[8][0] = dfs[8][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,0,0,0,0,0,0,1,0])
    if i == 9:
        dfs[9][0] = dfs[9][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,0,0,0,0,0,0,0,1])




for i in range(10):
    if i == 0:
        dfs_test[0][0] = dfs_test[0][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [1,0,0,0,0,0,0,0,0,0])
    if i == 1:
        dfs_test[1][0] = dfs_test[1][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,1,0,0,0,0,0,0,0,0])
    if i == 2:
        dfs_test[2][0] = dfs_test[2][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,1,0,0,0,0,0,0,0])
    if i == 3:
        dfs_test[3][0] = dfs_test[3][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,0,1,0,0,0,0,0,0])
    if i == 4:
        dfs_test[4][0] = dfs_test[4][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,0,0,1,0,0,0,0,0])     
    if i == 5:
        dfs_test[5][0] = dfs_test[5][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,0,0,0,1,0,0,0,0])
    if i == 6:
        dfs_test[6][0] = dfs_test[6][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,0,0,0,0,1,0,0,0])
    if i == 7:
        dfs_test[7][0] = dfs_test[7][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,0,0,0,0,0,1,0,0])
    if i == 8:
        dfs_test[8][0] = dfs_test[8][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,0,0,0,0,0,0,1,0])
    if i == 9:
        dfs_test[9][0] = dfs_test[9][0].replace([0, 1, 2, 3,4,5,6,7,8,9], [0,0,0,0,0,0,0,0,0,1])


#
#df0[0]
#
#df0[0] = df0[0].replace([0, 1, 2, 3,4,5,6,7,8,9], [1,0,0,0,0,0,0,0,0,0])
#df0_test[0] = df0_test[0].replace([0, 1, 2, 3,4,5,6,7,8,9], [1,0,0,0,0,0,0,0,0,0])
#
#df0[0]
#
#x0_train = np.array(df0.iloc[:,1:])
#x0_test = np.array(df0_test.iloc[:,1:])
#y0_train = np.array(df0[0])
#y0_test = np.array(df0_test[0])
#


x_train = {}
x_test =  {}
y_train = {}
y_test = {}

for i in range(10):
    x_train[i] = np.array(dfs[i].iloc[:,1:])
    x_test[i] =  np.array(dfs_test[i].iloc[:,1:])
    y_train[i] = np.array(dfs[i][0])
    y_test[i] = np.array(dfs_test[i][0])


y_test[6]



from sklearn.linear_model import LogisticRegression

log_m = LogisticRegression(solver = "lbfgs")

log_models = {}
y_pred = {}
y_pred_prob = {}
for i in range(10):
    log_models[i] = log_m.fit(x_train[i],y_train[i])
    y_pred[i] = log_m.predict(x_test[i])
    y_pred_prob[i] = log_m.predict_proba(x_test[i])





prob = {}
for i in range(10):
    prob[i] = y_pred_prob[i][:,1] 

unrelated_prob = pd.DataFrame(prob)

softmax(unrelated_prob.iloc[1,:])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0) 







soft_m = {}
for i in range(len(unrelated_prob)):
    soft_m[i] = softmax(unrelated_prob.iloc[i,:])
    


prob_dist = pd.DataFrame.from_dict(soft_m, orient='index')


max(prob_dist.iloc[1,:])


prob_dist.columns.get_values()[9]

predicted_cat = {}
for i in range(len(prob_dist)):
    predicted_cat[i] = prob_dist.iloc[i,:].idxmax(axis=1)



predicted_categories = pd.DataFrame.from_dict(predicted_cat, orient='index')


actual_categories = df_test[0]


data_for_matrix = pd.merge(actual_categories,predicted_categories,left_index = True , right_index= True)

data_for_matrix.columns

data_for_matrix = data_for_matrix.rename(columns = {"0_x":"Actual", "0_y":'Predicted'})


matrix = data_for_matrix.groupby(['Actual','Predicted']).size().unstack(fill_value=0)

print(matrix)






#
#
#log_m.fit(x0_train,y0_train)
#
#y0_pred = log_m.predict(x0_test)
#
#
#
#################
#y0_pred_prob = log_m.predict_proba(x0_test)
#
#
#logit_scores_y0 = log_m.predict_log_proba(x0_test)
#
#exps = [np.exp(i) for i in logit_scores_y0]
#
#
#
#from sklearn.metrics import confusion_matrix as cm
#confusion_matrix = cm(y0_test,y0_pred)
#print(confusion_matrix)
#
#log_m.predict_log_proba
#
#prob_for0 = log_m.predict_proba()
#