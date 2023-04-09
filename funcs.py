import streamlit as st
import pandas as pd
import random as rn

@st.cache_resource
def load_csv():
    p = 0.5
    url = "https://www.dropbox.com/s/6ba1n1kaiwv8ea3/Dataset_limpo.csv?raw=1"
    try:
        df = pd.read_csv(url, skiprows=lambda i: i>0 and rn.random() > p)
    except Exception as e:
        error_msg = "Erro ao carregar arquivo CSV"
        st.write(error_msg)
        raise Exception(error_msg)
    return df

@st.cache_resource
def load_csv2(): 
    url = "https://www.dropbox.com/s/f7lmv645avnajkd/steam.csv?raw=1"
    try:
        df_tags = pd.read_csv(url)
    except Exception as e:
        error_msg = "Erro ao carregar arquivo CSV"
        st.write(error_msg)
        raise Exception(error_msg)
    return df_tags
