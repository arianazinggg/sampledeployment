import numpy as np
import pickle
import streamlit as st
import tensorflow as tf
import pandas as pd





# load the model
#model = pickle.load(open('model.sav', 'rb'))
model = tf.keras.models.load_model('saved_model')


# load the scaler
scaler = pickle.load(open('scaler.sav', 'rb'))


def coffee_prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    scaled = scaler.transform(input_data_reshaped)

    prediction = model.predict(scaled)


    return f" You're expected yield is {prediction[0][0] * 1000 :.2f} kg/ha"



def main():

    # Website title
    st.set_page_config(page_title="Coffee Yield Prediction", 
                    page_icon= ':coffee:',
                    layout="wide",)
    st.title('A ***Brew-tiful*** Yield Forecast :coffee:')

    # Display image to the right of the text
    #col2.image(image_url, caption="Robusta Coffee Bean", use_column_width=True)


    st.markdown('### Input Fields')
# st.subheader("How to use the 'Explore' tab")
    #st.markdown('<p> The Explore tab allows you to <strong>select different models</strong> based on time intervals and test-train split data. <br> </p>', unsafe_allow_html=True)

    data = {
        "Term": ["Rainfall (mm)", "Surface Pressure (hPa)", "Dewpoint (°C)", "Relative Humidity (%)"],
        "Definition": ["Water falling from the atmosphere to the Earth's surface", 
                        "The measurement of air pressure at a specific location, adjusted as if that location were at sea level",
                        "Temperature at which air becomes saturated with moisture, leading to condensation — temperature at which the air gets so full of water vapor that it starts to make things wet",
                        "Relative humidity is a measure of how much moisture the air is holding compared to the maximum amount it could hold at a given temperature relative humidity —  how full the air is with moisture"],}

    df = pd.DataFrame(data)

    # Display the DataFrame as a table
    st.table(df)
    st.markdown('### Yield Prediction')
    # Getting input data from the user
    rain = st.number_input('Annual Total Rainfall (mm)', min_value = 0, max_value = None, value = None, placeholder='Enter any positive value') 
    sp = st.number_input('Annual Average Surface Pressure (hPa)', min_value = 900, max_value = 1060, value=None, placeholder='Enter a value between 900 and 1060') 
    dp = st.number_input('Annual Average Dew point (°C)', min_value = 10, max_value = 25, value = None, placeholder= 'Enter a value between 10 and 25') 
    rh = st.number_input('Annual Average Relative Humidity (%)', min_value = 1, max_value = 100, value = None, placeholder = 'Enter a value between 1 and 100' )

    # Code for prediction
    Yield = ''

    # Button for prediction
    if st.button('Predict Coffee Yield'):
        if None in [rain, sp, dp, rh]:
            st.error("Please fill in all fields.")
        
        else: 
            Yield = coffee_prediction([rain,sp,dp,rh])
    
    st.success(Yield)



if __name__ == '__main__':
    main()