# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:44:37 2020

@author: Shivam
"""
#library used
import pandas as pd
import numpy as np


""" loading the data and seperating the input and output data. Saving it as numpy array """
df_og = pd.read_excel("bcd.xlsx", set_index = 0,axis=1,header = None )
df_sub = df_og.drop(0,axis =1)
df_bcd = df_sub.iloc[:10,:5]
df_out = df_sub.iloc[:,6:]
df_bcd2 = df_bcd.copy(deep=True)
np_bcd = np.array(df_bcd)
np_bcd2 = np.array(df_bcd2)
np_out = np.array(df_out)

""" loop that will create required truth table """  
final = []
for i in range(len(np_bcd)):
    for j in range(len(np_bcd2)):
        k=0
        k =i+j
        final.append(np_bcd[i,:])
        final.append(np_bcd2[j,:])
        final.append(np_out[k,:])

 """converting the truth table in np array """
final_np = np.array(final)    
df_np = np.reshape(final_np,1500)
df = np.reshape(df_np, (-1,15)
df_tt = pd.DataFrame(df)
df_tt = df_tt.drop([0, 5], axis=1)

""" splitting the data into inpit and output as numpy array """
X = df_tt.iloc[:,:8]
Y = df_tt.iloc[:,8:]

x_np = np.array(X)

y_np = np.array(Y)




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#library used
import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


NN_model = Sequential()
# The Input Layer :
NN_model.add(Dense(8, activation='sigmoid'))
# The Hidden Layers :
NN_model.add(Dense(7,activation='sigmoid'))
#NN_model.add(Dropout(0.2))
# The Output Layer :
NN_model.add(Dense(5,activation='sigmoid'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
NN_model.summary()

# Fitting the ANN to the Training set
model_history=NN_model.fit(X_input, y_output, nb_epoch = 5000)


