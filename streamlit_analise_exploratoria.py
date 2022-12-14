
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import streamlit as st

cols = ['appid', 'name', 'release_date', 'english', 'developer', 'publisher', 'platforms', 'required_age', 'categories', 'genres', 'steamspy_tags', 'achievements', 'positive_ratings', 'negative_ratings', 'average_playtime', 'median_playtime', 'owners', 'price']
dataset = pd.read_excel(io = 'steam.xlsx', engine = 'openpyxl', sheet_name= 'steam1', skiprows= 2, usecols='A:R', nrows=500, header=None, names=cols)
#print(dataset)
df = pd.DataFrame(dataset)
#print(df)
matrizCorr = df.corr()
#print(matrizCorr)

fig, ax = plt.subplots()
sb.set(font_scale=0.7)  
sb.heatmap(matrizCorr, annot=True, ax=ax)
st.write(fig)

