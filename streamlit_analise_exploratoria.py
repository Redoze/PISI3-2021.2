
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title='EDA Steam Dataset', 
                   layout="wide")

cols = ['ID', 'Nome', 'Data de lançamento', 'Em inglês?', 'Dev', 'Publicador', 'Plataformas', 'Classificação indicativa', 'Categorias', 'Gêneros', 'Tags de comunidade', 'Troféus', 'Avaliações positivas', 'Avaliações negativas', 'Tempo de jogo médio', 'Tempo de jogo mediano', 'Quantidade de jogadores', 'Preço']
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

