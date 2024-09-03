
import sys
#sys.path.append('/Volumes/TOSHIBA EXT/Maestria/Programs/Machine Learning/End-to-End Housing/')
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle as pkl
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from joblib import dump, load
from custom_transformers import CombinedAttributesAdder



print(sys.path)

# from exports import custom_transformers

#full_pipeline = joblib.load('/Volumes/TOSHIBA EXT/Maestria/Programs/Machine Learning/End-to-End Housing/exports/full_pipeline.pkl')

pipiline_path = 'exports/full_pipeline.pkl'
with open(pipiline_path, 'rb') as file1:
    print(file1.read(100))  
try:
    pipeline = joblib.load(pipiline_path)
    print("pipeline loaded successfully!")
except Exception as e:
    print("Failed to load pipeline:", e)

model_path = 'exports/model.pkl'

with open(model_path, 'rb') as file:
    print(file.read(100))  
try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print("Failed to load model:", e)

#print(pipeline)

st.set_page_config(page_title='Housing Price Prediction', page_icon=':house:', layout='wide')


st.markdown("""
    <style>
    body {
        background-color: white;
    }
    .title {
        text-align: center;
        color: #6C86BF;
        margin-bottom: 20px;
    }
    div.stButton > button:first-child {
        display: block;
        margin: 0 auto;
        background-color: #6C86BF;
        color: white;
        padding: 10px 24px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)




st.markdown('<h1 class="title">Housing Price Prediction</h1>', unsafe_allow_html=True)

st.markdown('<h2 style="text-align: center;"> Input features </h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)


with col1:
    #Unnamed = st.number_input('Enter a Unnamed (0 to 21000)', min_value=0.0, max_value=21000.0, value=0.0, step=1.0)
    longitude = st.slider('Enter a longitude (-125 to -114)', min_value=-124.3500, max_value=-114.3100, value=-120.0, step=0.05)
    latitude = st.slider('Enter a latitude (30 to 45)', min_value=30.0, max_value=45.0, value=32.0, step=0.05)
    housing_median_age = st.slider('Enter a housing median age (1 to 100)', min_value=1.0, max_value=100.0, value=35.0, step=1.0, format= '%.0f')
    total_rooms = st.slider('Enter a total rooms (1 to 50000)', min_value=1.0, max_value=50000.0, value=1000.0, step=1.0, format= '%.0f')
with col2:
    total_bedrooms = st.slider('Enter a total bedrooms (1 to 7000)', min_value=1.0, max_value=7000.0, value=1000.0, step=1.0, format= '%.0f')
    population = st.slider('Enter a population (1 to 50000)', min_value=1.0, max_value=50000.0, value=1000.0, step=1.0, format= '%.0f')
    households = st.slider('Enter a households (1 to 10000)', min_value=1.0, max_value=10000.0, value=1000.0, step=1.0, format= '%.0f')
    median_income = st.slider('Enter a median income (0 to 15)', min_value=0.0, max_value=20.0, value=5.0, step=1.0)

ocean_proximity = st.selectbox('Enter a ocean proximity', ('<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'))

mapdata = pd.DataFrame({
    'lat': [latitude],
    'lon': [longitude]
})

st.markdown("""
<style>
div.stButton > button:first-child {
    display: block;
    margin: 0 auto;
}
</style>""", unsafe_allow_html=True)


col3, col4, col5 = st.columns([1,6,1])


if st.button('Predict'):
    with col4:
        input_data = pd.DataFrame(
    
        {'longitude': [longitude], 'latitude': [latitude], 'housing_median_age': [housing_median_age],
         'total_rooms': [total_rooms], 'total_bedrooms': [total_bedrooms], 'population': [population],
         'households': [households], 'median_income': [median_income], 'ocean_proximity': [ocean_proximity]},
         index=[0]
    )
        st.write(input_data)

    pipelined_data = pipeline.transform(input_data)

    prediction = model.predict(pipelined_data)

    col6, col7, col8 = st.columns(3)

    with col7:
        #st.write('The predicted housing price is $', int(prediction[0]))
        st.markdown('<h2 style="text-align: center;"> The predicted housing price is $' + str(int(prediction[0])) + ' </h2>', unsafe_allow_html=True)


st.map(mapdata)

