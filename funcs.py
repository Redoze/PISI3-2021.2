import streamlit as st
import pandas as pd
import random as rn

# Load data
@st.cache_data

def load_csv():
    df = pd.read_csv("Dataset_limpo.csv", index_col=0)
    return df