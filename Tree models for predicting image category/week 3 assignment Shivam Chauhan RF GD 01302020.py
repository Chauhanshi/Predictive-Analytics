

#library used
import pandas as pd
import os
import numpy as np


#Changing the working directory
os.getcwd()
#os.chdir("D:")
os.chdir("C:/Users\Shivam\OneDrive - Northeastern University\STUDY\ALY 6020 - Predictive\week 2")


# defining the column names
name=[]
for i in range(785):
    name.append(i)
    
#Loading the data and looking into the data type
df = pd.read_csv("fashion_train.csv" , names = name)
df_test = pd.read_csv("fashion_test.csv", names = name)

#df.columns
df.shape


#creating traing data set for 10 different models for each fashion item category
dfs = {}
for i in range(10):
    dfs[i] = df.copy(deep=True)

#creating test data set for 10 different models for each fashion item category
dfs_test = {}
for i in range(10):
    dfs_test[i] = df_test.copy(deep=True)
    
    
    



#preparing the train data
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



#preparing the test data
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





#creating empty dictionary that can store test and train data. 
x_train = {}
x_test =  {}
y_train = {}
y_test = {}

#Splitting the data into test and train for each item category and storing it in dictionary
for i in range(10):
    x_train[i] = np.array(dfs[i].iloc[:,1:])
    x_test[i] =  np.array(dfs_test[i].iloc[:,1:])
    y_train[i] = np.array(dfs[i][0])
    y_test[i] = np.array(dfs_test[i][0])

