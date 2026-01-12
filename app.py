
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and preprocessed dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Input widgets for each feature

# Manufacturer
company = st.selectbox('Brand', df['Manufacturer'].unique())

# Category
category = st.selectbox('Category', df['Category'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', sorted(df['RAM'].unique()))

# Weight
weight = st.number_input('Weight of the Laptop (in kg)', min_value=0.1, max_value=5.0, value=1.5)

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
if touchscreen == 'Yes':
    touchscreen_val = 1
else:
    touchscreen_val = 0

# IPS Panel
ips = st.selectbox('IPS Panel', ['No', 'Yes'])
if ips == 'Yes':
    ips_val = 1
else:
    ips_val = 0

# Screen Size
screen_size = st.number_input('Screen Size (in inches)', min_value=10.0, max_value=20.0, value=15.6)

# Resolution
resolution = st.selectbox('Screen Resolution', sorted(df['X'].astype(str) + 'x' + df['Y'].astype(str), key=lambda x: int(x.split('x')[0]), reverse=True))

X_res = int(resolution.split('x')[0])
Y_res = int(resolution.split('x')[1])
ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

# Processor Name
processor = st.selectbox('CPU Brand', df['ProcessorName'].unique())

# SSD
ssd = st.selectbox('SSD (in GB)', sorted(df['SSD'].unique().astype(int)))

# HDD
hdd = st.selectbox('HDD (in GB)', sorted(df['HDD'].unique().astype(int)))

# GPU Brand
gpu = st.selectbox('GPU Brand', df['GPUName'].unique())

# Operating System
os = st.selectbox('Operating System', df['Operating System'].unique())

# Operating System Version
os_version = st.selectbox('Operating System Version', df['Operating System Version'].unique())


if st.button('Predict Price'):
    # Create a DataFrame for prediction
    query = pd.DataFrame([[
        company,
        category,
        screen_size,
        ram,
        os,
        os_version,
        weight,
        touchscreen_val,
        ips_val,
        ppi,
        processor,
        gpu,
        float(ssd),
        float(hdd)
    ]],
    columns=['Manufacturer', 'Category', 'Screen Size', 'RAM', 'Operating System',
             'Operating System Version', 'Weight', 'Touchscreen', 'IPS', 'ppi',
             'ProcessorName', 'GPUName', 'SSD', 'HDD'])

    # Predict the price
    predicted_price = np.expm1(pipe.predict(query)[0])

    st.title("The predicted price of this configuration is ₹ " + str(int(predicted_price)))
