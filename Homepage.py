import streamlit as st

st.set_page_config(page_title="Coffee Yield Prediction", 
                    page_icon=":coffee:",
                    layout="wide",
                    initial_sidebar_state="expanded",
)

st.title("Coffee Yield Prediction")

#reading the csv file to visualizations

#---tabs----

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


tab1, tab2= st.tabs(["🗃 Data","📈 Chart" ])

with tab1:
    st.header('Yield & Weather Data')
    df= pd.read_csv('Final Yield_Weather (Mindanao).csv')
    df.rename(columns = {'weathercode (wmo code)':'Weather Code', 'temperature_2m (°C)' : 'Temperature (°C)', 
                          'relativehumidity_2m (%)': 'Relative Humidity (%)', 'dewpoint_2m (°C)':'Dew point(°C)', 
                          'rain (mm)':'Rain (mm)', 'surface_pressure (hPa)':'Surface Pressure(hPa)', 'cloudcover (%)':'Cloud Cover(%)', 
                          'et0_fao_evapotranspiration (mm)': 'Evapotranspiration (mm)', 'vapor_pressure_deficit (kPa)': 'Vapor Pressure Deficit(kPa)', 
                           'Wind_Direction':'Wind Direction', 'Wind_Speed_(km/h)': 'Wind Speed (km/h)', 'windgusts_10m (km/h)': 'Wind Gusts(km/h)', 
                           'soil_temperature_28_to_100cm (°C)':'Soil Temperature (°C)', 'soil_moisture_28_to_100cm (m³/m³)':'Soil Moisture (m³/m³)', 'direct_radiation (W/m²)':'Direct Radiation (W/m²)',
                            'diffuse_radiation (W/m²)':'Diffuse Radiation (W/m²)'}, inplace = True)
    st.dataframe(df)

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df(df)
    
    st.download_button(
        "Download CSV file",
        csv,
        "Yield & Weather of Mindanao.csv",
        "text/csv",
        key='download-csv'
        )
    

with tab2:
    import altair as alt
    #st.header('Line Chart')
    data = {
    'y': ['Weather code', 'Temperature (°C)',
       'Relative Humidity (%)', 'Dew point(°C)', 'Rain (mm)',
       'Sirface Pressure(hPa)', 'Cloud Cover(%)',
       'Evapotranspiration (mm)', 'Vapor Pressure Deficit(kPa)',
       'Wind Direction', 'Wind Speed (km/h)', 'Wind Gusts(km/h)',
       'Soil Temperature (°C)',
       'Soil Moisture (m³/m³)', 'Direct Radiation (W/m²)',
       'Diffuse Radiation (W/m²)', 'Yield (mt/ha)']}
    
    col1, col2 = st.columns(2)

    with col1:
        #st.header("Select a parameter")
        selected_variable = st.selectbox('Select a variable', df.columns[3:], key="option1")

    with col2:
        #st.header('By Location')
        data = {
        'Location': ['AGUSAN DEL NORTE', 'AGUSAN DEL SUR', 'BASILAN', 'BUKIDNON', 'CAMIGUIN', 'COTABATO', 'DAVAO CITY', 'DAVAO DE ORO', 'DAVAO DEL NORTE', 'DAVAO DEL SUR', 'DAVAO ORIENTAL', 'LANAO DEL NORTE', 'LANAO DEL SUR', 
                    'MAGUINDANAO', 'MISAMIS OCCIDENTAL', 'MISAMIS ORIENTAL', 'SARANGANI', 'SOUTH COTABATO', 'SULTAN KUDARAT', 'SULU', 'SURIGAO DEL SUR', 'ZAMBOANGA DEL NORTE','ZAMBOANGA DEL SUR', 'ZAMBOANGA SIBUGAY' ]}
        selected_location = st.selectbox('Select a Location', df['Location'].unique(), key="option2")
        selected_data = df[(df['Location'] == selected_location)]
        #st.write(selected_data)
    
    df['Year'] = df['Year'].astype(int) 
    #chart_data = df.groupby('Year')[[selected_variable, selected_location]].sum().reset_index()
    location_data = df[df['Location'] == selected_location]
    chart_data = location_data.groupby('Year')[selected_variable].sum().reset_index() 
    line_chart = alt.Chart(chart_data).mark_line().encode(
    x='Year',y=selected_variable).properties(width=800)
    st.altair_chart(line_chart)