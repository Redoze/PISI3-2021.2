import streamlit as st
import pandas as pd
import random as rn

# Load data
@st.cache_data
def load_csv():
    p=0.05
    df = pd.read_csv("Dataset_limpo.csv", skiprows=lambda i: i>0 and rn.random() > p)
    return df

@st.cache_data
def load_csv2():
    p=0.05
    df_tags = pd.read_csv("steam.csv", skiprows=lambda i: i>0 and rn.random() > p)
    return df_tags
