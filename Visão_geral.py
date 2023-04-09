import pandas as pd
import seaborn as sb
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import random as rn
from wordcloud import WordCloud
import altair as alt
from funcs import *

#cd 'C:\Users\josef\Desktop\Projeto 3'
#streamlit run Visão_geral.py --server.port 80

st.set_page_config(
    page_title="Análise de sentimentos em avaliações de jogos na Steam",
    page_icon="✅",
    layout="wide",
)

df = load_csv()
df_tags = load_csv2()
df_merged = pd.merge(df, df_tags, left_on=["app_id", "app_name"], right_on=["appid", "name"])
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  -  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

st.title("Análise de sentimentos em avaliações de jogos na Steam")

st.markdown("---")

ocultar_df = st.sidebar.checkbox('Ocultar conjunto de dados')

if ocultar_df:
    st.sidebar.write('Conjunto de dados ocultado')

else:
    st.header('Visão geral do conjunto de dados')
    st.text("")
    st.dataframe(df_merged)
    st.caption('review_score:  1 = review positiva, review_score: -1 = review negativa, review_votes:  1 = review recomendada, review_votes:  0 = review sem recomendação ou negativada')
    st.text("")

graph_options = ["Histograma de sentimentos","Histograma de contagem de reviews recomendados por sentimento","Gráfico de pizza de distribuição de sentimentos"]#"Histograma dos 10 jogos com mais reviews",

st.sidebar.subheader("Use o seletor para analisar todo do conjunto de dados:")
st.text("")
selected_chart = st.sidebar.selectbox('',graph_options)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  -  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# if selected_chart == "Histograma dos 10 jogos com mais reviews":
#     st.subheader("Histograma dos 10 jogos com mais reviews") 
#     dados_jogos = df.groupby(["app_id", "app_name", "review_score"]).agg({"review_text": "count"}).reset_index()
#     dados_jogos = dados_jogos.rename(columns={"review_text": "review_count"})
#     top10_jogos = dados_jogos.groupby(["app_id", "app_name"]).agg({"review_count": "sum"}).sort_values("review_count", ascending=False).head(10).reset_index()
#     dados_jogos = dados_jogos[dados_jogos["app_id"].isin(top10_jogos["app_id"])]
#     barras_horizontal = px.bar(dados_jogos, y="app_name", x="review_count", color="review_score", orientation="h",
#                 hover_data=["review_count"], category_orders={"app_name": top10_jogos["app_name"]})
#     barras_horizontal.update_layout(
#     #title="Os 10 jogos com mais reviews",
#     xaxis_title="Número de reviews",
#     yaxis_title="Jogo",
#     legend_title="Sentimento",
#     width=1000,
#     height=600,
#     margin=dict(l=50, r=50, t=80, b=80),
#     hovermode="closest")
#     st.plotly_chart(barras_horizontal)
#     st.write("Representação gráfica dos 10 jogos com mais reviews, a sua distribuição de sentimento e o número de reviews")

if selected_chart == "Histograma de sentimentos":
    st.subheader("Histograma de sentimentos")
    st.write('')
    histograma_sentimentos = alt.Chart(df).mark_bar().encode(
    x=alt.X('review_score', scale=alt.Scale(domain=[-1, 1]), 
            axis=alt.Axis(tickCount=2, values=[-1,1])),
    y='count()',
    color=alt.Color('review_score', legend=None,
        scale=alt.Scale(domain=[-1,1], range=['#FF4136', '#2ECC40']))
    ).properties(width=600, height=400)
    st.altair_chart(histograma_sentimentos)
    st.write("Representação gráfica da distribuição de sentimentos em reviews de jogos da Steam")

elif selected_chart == "Histograma de contagem de reviews recomendados por sentimento":
    st.subheader("Histograma de contagem de reviews recomendados por sentimento")
    sentiment_votes = df.groupby(['review_score', 'review_votes'])['app_id'].count().unstack('review_votes')
    barras_empilhadas = px.bar(sentiment_votes, barmode='stack', labels={'value': 'Contagem', 'review_score': 'Sentimento'})
    barras_empilhadas.update_layout(#title="Contagem de reviews recomendados e não recomendados por sentimento"
    )
    st.plotly_chart(barras_empilhadas)
    st.write("Representação gráfica da contagem de reviews recomendadas e não recomendadas por sentimento")

elif selected_chart == "Gráfico de pizza de distribuição de sentimentos":
    st.subheader("Gráfico de pizza de distribuição de sentimentos")
    pizza_chart = px.pie(df, values='review_votes', names='review_score', color='review_score')
    pizza_chart.update_layout(
    # title="Distribuição de sentimentos",
    legend_title="Sentimento",
    width=1000,
    height=600)
    st.plotly_chart(pizza_chart)
    st.write("Representação gráfica da proporção de sentimentos positivos e negativos nas reviews")
