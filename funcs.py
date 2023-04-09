import streamlit as st
import pandas as pd
import random as rn

# Load data
@st.cache_data
def load_csv():
    url = "https://www.dropbox.com/s/6ba1n1kaiwv8ea3/Dataset_limpo.csv?raw=1" #Dataset_limpo.csv
    df = pd.read_csv(url)
    return df

@st.cache_data
def load_csv2(): 
    url = "https://www.dropbox.com/s/f7lmv645avnajkd/steam.csv?raw=1" #steam.csv
    df_tags = pd.read_csv(url)
    return df_tags
