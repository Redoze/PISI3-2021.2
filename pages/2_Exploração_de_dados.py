import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random as rn
import altair as alt
import plotly.express as px
from funcs import *
 
st.set_page_config(
    page_title="Exploração de dados",
    page_icon="🔎",
    layout="wide",
)

df = load_csv()
df_tags = load_csv2()
df_merged = pd.merge(df, df_tags, left_on=["app_id", "app_name"], right_on=["appid", "name"])

st.title("Explorando os dados dos jogos")
st.markdown("---")
st.sidebar.subheader("Use os filtros para exploração de dados:")

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  -  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#Define os itens a serem selecionados na lista dropdown
game_options = df["app_name"].unique()
review_options = [-1, 1]
graph_options = ["Nuvem de palavras", "Histograma das 10 palavras mais frequentes","Histograma de sentimentos","Histograma de contagem de reviews recomendados por sentimento","Gráfico de pizza de distribuição de sentimentos"]

#Usa o multiselect pra definir as opções
selected_games = st.sidebar.multiselect("Selecione o(s) jogo(s)", game_options)
selected_reviews = st.sidebar.multiselect("Selecione o tipo de review", review_options)
selected_graph = st.sidebar.selectbox("Selecione o gráfico", graph_options)
 
#cria um dataframe de dados filtrados baseados nas opções selecionadas
filtered_data = df[(df["app_name"].isin(selected_games)) & (df["review_score"].isin(selected_reviews))]

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  -  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

if filtered_data.shape[0] == 0:
    st.write("Selecione um jogo e um tipo de review na barra lateral")
else:
    #Cria os gráficos baseado no que foi escolhido no dropdown
    if selected_graph == "Histograma das 10 palavras mais frequentes":
        st.subheader('Histograma das 10 palavras mais frequentes')
        st.write('')
        #Dataframe que contém a contagem de cada palavra nas review s
        word_counts = filtered_data["review_text"].str.split(expand=True).stack().value_counts().reset_index()
        word_counts.columns = ["word", "count"]
        
        #ordena os valores pela contagem em ordem descendente e seleciona apenas as top 10 palavras mais usadas
        top_words = word_counts.sort_values("count", ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        ax = top_words.plot(kind="bar", x="word", y="count", rot=45)
        ax.set_xlabel("Palavras", fontsize=16)
        ax.set_ylabel("Contagem", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12, pad=8)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        st.write('Representação gráfica das 10 palavras mais frequentes')

    elif selected_graph == "Nuvem de palavras":
        st.subheader("Nuvem de palavras")
        st.write('')
        text = " ".join(review for review in filtered_data.review_text)
        wordcloud = WordCloud(max_words=100, background_color="white").generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt.gcf())
        st.write('Representação em formato de nuvem das palavras mais frequentes')

    elif selected_graph == "Histograma de sentimentos":
        st.subheader("Histograma de sentimentos")
        st.write('')
        st.write('Mantenha selecionado os dois tipos de reviews para este gráfico')
        st.write('')
        histograma_sentimentos = alt.Chart(filtered_data).mark_bar().encode(
        x=alt.X('review_score', scale=alt.Scale(domain=[-1, 1]), 
                axis=alt.Axis(tickCount=2, values=[-1,1])),
        y='count()',
        color=alt.Color('review_score', legend=None,
            scale=alt.Scale(domain=[-1,1], range=['#FF4136', '#2ECC40']))
        ).properties(width=600, height=400)
        st.altair_chart(histograma_sentimentos)
        st.write("Representação gráfica da distribuição de sentimentos nos jogos selecionados")

    elif selected_graph == "Histograma de contagem de reviews recomendados por sentimento":
        st.subheader("Contagem de reviews recomendados e não recomendados por sentimento")
        sentiment_votes = filtered_data.groupby(['review_score', 'review_votes'])['app_id'].count().unstack('review_votes')
        barras_empilhadas = px.bar(sentiment_votes, barmode='stack', labels={'value': 'Contagem', 'review_score': 'Sentimento'})
        barras_empilhadas.update_layout(title="Utilize os dois tipos de review para uma melhor exploração")
        st.plotly_chart(barras_empilhadas)
        st.write("Representação gráfica da contagem de reviews recomendadas e não recomendadas por sentimento")

    elif selected_graph == "Gráfico de pizza de distribuição de sentimentos":
        st.subheader('Distribuição de sentimentos')
        pizza_chart = px.pie(filtered_data, values='review_votes', names='review_score', color='review_score')
        pizza_chart.update_layout(
        title="Mantenha selecionado os dois tipos de reviews para este gráfico",
        legend_title="Sentimento",
        width=1000,
        height=600)
        st.plotly_chart(pizza_chart)
        st.write("Representação gráfica da proporção de sentimentos positivos e negativos nas reviews")

