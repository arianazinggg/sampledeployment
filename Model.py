import numpy as np
import pickle
import streamlit as st
import tensorflow as tf



# load the model
#model = pickle.load(open('model.sav', 'rb'))
model = tf.keras.models.load_model('model.sav')


# load the scaler
scaler = pickle.load(open('scaler.sav', 'rb'))


def coffee_prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    scaled = scaler.transform(input_data_reshaped)

    prediction = model.predict(scaled)

    return f"Predicted output: {prediction[0][0]:.2f}"



def main():

    # Website title
    st.title('Coffee Yield Prediction Web App')

    # Getting input data from the user
    rain = st.number_input('Annual Total Rainfall (mm)') 
    sp = st.number_input('Annual Average Surface Pressure (hPa)') 
    dp = st.number_input('Annual Average Dewpoint (°C)') 
    rh = st.number_input('Annual Average Relative Humidity (%)')

    # Code for prediction
    Yield = ''

    # Button for prediction
    if st.button('Predict Coffee Yield'):
        Yield = coffee_prediction([rain,sp,dp,rh])
    
    st.success(Yield)



if __name__ == '__main__':
    main()