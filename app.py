import streamlit as st 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from numpy import loadtxt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import time
from keras import metrics

st.title('Coffee Yield Prediction')
st.subheader('Exploratory Data Analysis')

whole_data=pd.read_csv('Final Yield_Weather (Mindanao).csv')


whole_data['Date'] = pd.to_datetime(whole_data["Date"], format='%Y-%m-%d')
whole_data.sort_values(by=["Date"], inplace=True)

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LinearRegression

def MLR_Model(X_train,y_train):
    mlr = LinearRegression()
    mlr.fit(X_train,y_train)
    return mlr

def ANN_Model (initial_weights,bias): 
    model = Sequential()
    model.add(Input(shape = 4,))
    model.add(Dense(1, activation = 'relu', name = 'hidden'))
    model.add(Dense(1,activation = 'linear'))


    #initialize weights and bias in hidden layer
    hidden_layer = model.get_layer('hidden')
    #layer.set_weights([weights_array, bias_array])
    hidden_layer.set_weights([np.array([initial_weights]).reshape((4, 1)), np.array([bias])])


    #compile 
    model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
    return model

X = whole_data[['weathercode (wmo code)', 'temperature_2m (°C)',
       'relativehumidity_2m (%)', 'dewpoint_2m (°C)', 'rain (mm)',
       'surface_pressure (hPa)', 'cloudcover (%)',
       'et0_fao_evapotranspiration (mm)', 'vapor_pressure_deficit (kPa)',
       'Wind_Direction', 'Wind_Speed_(km/h)', 'windgusts_10m (km/h)',
       'soil_temperature_28_to_100cm (°C)',
       'soil_moisture_28_to_100cm (m³/m³)', 'direct_radiation (W/m²)',
       'diffuse_radiation (W/m²)']]
y = whole_data['Yield (mt/ha)']

from keras import backend
 
def rmse(y_true, y_pred):
 return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LinearRegression
def MLR_Model(X_train,y_train):
    mlr = LinearRegression()
    mlr.fit(X_train,y_train)
    return mlr

def ANN_Model (initial_weights,bias): 
    model = Sequential()
    model.add(Input(shape = 16,)) # number of input features 
    model.add(Dense(1, activation = 'relu', name = 'hidden')) # one neuron in hidden layer, relu activation
    model.add(Dense(1,activation = 'linear')) # one neuron in output layer, linear activation


    #initialize weights and bias in hidden layer
    hidden_layer = model.get_layer('hidden')
    #layer.set_weights([weights_array, bias_array])
    hidden_layer.set_weights([np.array([initial_weights]).reshape((16, 1)), np.array([bias])])


    #compile 
    model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),metrics = [metrics.mean_absolute_error,rmse])
    return model

start_time = time.time()

#initializing KFold object
kfold = KFold(n_splits=10,shuffle=True, random_state=42)
mae_scores = []
r2_scores = []

for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # Predictors for train & test
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # Target for train, test

    # scaling fetaure sets 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mlr_model = MLR_Model(X_train,y_train) #perform MLR
    ann_model = ANN_Model(mlr_model.coef_,mlr_model.intercept_) #Intercept and coeff of MLR will be initailzied as ANN weights
    ann_model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=0) #fit ann model to data
    y_pred = ann_model.predict(X_test) #use ann model to predict test set
    mae = mean_absolute_error(y_test, y_pred) #measure mean absolute value
    print(f"MAE: {mae}")
    r2 = r2_score(y_test,y_pred) #measure r-squared
    print(f"R2: {r2}")
    print(f"Intercept: {mlr_model.intercept_}")
    print(f"Coefficients: {list(zip(mlr_model.coef_))}")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# streamlit style

page = st.sidebar.selectbox("""
Hello there! I'll guide you!
                            
                            Please select model""", ["HOME", "About the Model"])

