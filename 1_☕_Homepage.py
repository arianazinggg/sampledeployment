import streamlit as st

st.set_page_config(page_title="Coffee Yield Prediction", 
                    page_icon="🧊",
                    layout="wide",
                    initial_sidebar_state="expanded",
)

st.title("Coffee Yield Prediction")
st.sidebar.success("Select a page above.")


#reading the csv file to visualizations

#---tabs----

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


tab1, tab2, tab3 = st.tabs(["🗃 Data", "Filters", "📈 Chart" ])

with tab1:
    st.header('Yield & Weather Dataset [Mindanao]')
    df= pd.read_csv('/workspaces/sampledeployment/Final Yield_Weather (Mindanao).csv')
    st.dataframe(df)

with tab2:
    st.header('By Location')
    data = {
    'Location': ['AGUSAN DEL NORTE', 'AGUSAN DEL SUR', 'BASILAN', 'BUKIDNON', 'CAMIGUIN', 'COTABATO', 'DAVAO CITY', 'DAVAO DE ORO', 'DAVAO DEL NORTE', 'DAVAO DEL SUR', 'DAVAO ORIENTAL', 'LANAO DEL NORTE', 'LANAO DEL SUR', 
                 'MAGUINDANAO', 'MISAMIS OCCIDENTAL', 'MISAMIS ORIENTAL', 'SARANGANI', 'SOUTH COTABATO', 'SULTAN KUDARAT', 'SULU', 'SURIGAO DEL SUR', 'ZAMBOANGA DEL NORTE','ZAMBOANGA DEL SUR', 'ZAMBOANGA SIBUGAY' ]}
    selected_location = st.sidebar.selectbox('Select a Location', df['Location'].unique())
    selected_data = df[(df['Location'] == selected_location)]
    st.write(selected_data)

    st.header('By Year')
    data = {
    'Year': [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],}
    selected_year = st.sidebar.selectbox('Select a Year', df['Year'].unique())
    selected_data = df[(df['Year'] == selected_year)]
    st.write(selected_data)

    
import altair as alt

with tab3:
    st.header('Line Chart')
    data2 = {
    'y': ['weathercode (wmo code)', 'temperature_2m (°C)',
       'relativehumidity_2m (%)', 'dewpoint_2m (°C)', 'rain (mm)',
       'surface_pressure (hPa)', 'cloudcover (%)',
       'et0_fao_evapotranspiration (mm)', 'vapor_pressure_deficit (kPa)',
       'Wind_Direction', 'Wind_Speed_(km/h)', 'windgusts_10m (km/h)',
       'soil_temperature_28_to_100cm (°C)',
       'soil_moisture_28_to_100cm (m³/m³)', 'direct_radiation (W/m²)',
       'diffuse_radiation (W/m²)', 'Yield (mt/ha)']}
    
    selected_variable = st.selectbox('Select a variable', df.columns[2:])

    chart_data = df.groupby('Year')[selected_variable].sum().reset_index()
    line_chart = alt.Chart(chart_data).mark_line().encode(
    x='Year',
    y=selected_variable
    ).properties(width=800)
    st.altair_chart(line_chart)







