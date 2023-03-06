import pandas as pd
import seaborn as sb
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import random as rn
from wordcloud import WordCloud
import mplcursors

p = 0.02
df = pd.read_csv("Dataset_limpo.csv",
                    skiprows=lambda i: i>0 and rn.random() > p)

st.set_page_config(
    page_title="Análise de Popularidade de Jogos na Plataforma Steam",
    page_icon="✅",
    layout="wide",
)

st.sidebar.header("Análise de sentimentos em avaliações de jogos na Steam")

st.sidebar.header("Use os filtros para explorar os dados de um determinado jogo:")

jogo = st.sidebar.multiselect(
        "Selecione um Jogo:",
        options=df["app_name"].unique())
nota = st.sidebar.multiselect(
        "Selecione um Jogo:",
        options=df["review_score"].unique())

df_selection = df.query(
    "app_name == @jogo & review_score == @nota")

st.markdown("---")

nota_p_jogo = (
    df.groupby(by=['review_score']).count().sort_values(by="app_name"))


#:::::::::::::::::::::::::::::: GRÁFICOS ::::::::::::::::::::::::::::::

#Gráfico de Barra
fig_notas = px.bar(nota_p_jogo,
    x=nota_p_jogo.index,
    y="app_name",
    orientation="v",
    title="<b>Notas Positivas ou Negativas por Jogo",
    color_discrete_sequence=["#0083B8"]*len(nota_p_jogo),
    template="plotly_white",
    labels={
        "app_name":"Nº de Jogos (Milhares)",
        "review_score":"Nota (Positiva/Negativa)"},
    )

#Wordcloud
palavras = " ".join(str(q) for q in df.review_text)
fig_wordcloud = WordCloud(
            max_font_size = 100,
            max_words = 100,
            width = 1000,
            height = 600).generate(palavras)

#Histograma de contagem de palavras
df['contagem_palavras'] = df['review_text'].str.split().str.len() 
fighist = plt.figure()
plt.hist(df['contagem_palavras'], bins=30, edgecolor='black')
plt.xlabel('Contagem de Palavras')
plt.ylabel('Frequência')

#Bubble chart
game_sentiment_counts = df.groupby(['app_name', 'review_score'])['review_text'].count().reset_index()

sb.set_style("whitegrid")
sb.set_palette("Set1")
plt.figure(figsize=(10,8))
sb.scatterplot(data=game_sentiment_counts, x='review_text', y='review_score', size='review_text', hue='review_score', sizes=(100,1000), legend=False)

cursor = mplcursors.cursor(plt.gca())
@cursor.connect("add")
def on_add(sel):
    x, y, app_name, review_score, count = sel.target
    st.write(f'{app_name}\nSentiment: {review_score}\nCount: {count}')

plt.title("Distribuição de sentimento por jogo", fontsize=16)
plt.xlabel("Número de reviews", fontsize=14)
plt.ylabel("Score de sentimento", fontsize=14)

st.pyplot(plt)

#:::::::::::::::::::::::::::::: LAYOUT ::::::::::::::::::::::::::::::

st.title("Análise Global do Dataset")

#Chamada dos gráficos
fig, ax = plt.subplots(figsize = (20, 12))
ax.imshow(fig_wordcloud)
plt.axis("off")
st.pyplot(fig)
st.plotly_chart(fig_notas)
st.pyplot(fighist)