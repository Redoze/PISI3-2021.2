import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import random as rn
from wordcloud import WordCloud
import plotly.graph_objects as go
from funcs import *

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Exploração de dados",
    page_icon="🔎",
    layout="centered",
)

def build_header():
    st.title("Exploração de dados dos jogos na Steam")
    st.markdown("---")

def build_body():
    # Carrega os dataframes
    df = load_csv()
    df_tags = load_csv2()
    df_merged = pd.merge(df, df_tags, left_on=["app_id", "app_name"], right_on=["appid", "name"])

    st.sidebar.subheader("Use os filtros para exploração de dados:")

    # Define os itens a serem selecionados na lista dropdown
    game_options = df["app_name"].unique()
    review_options = {"Negativa": -1, "Positiva": 1}
    graph_options = ["Nuvem de palavras", "Histograma das 10 palavras mais frequentes", "Histograma de sentimentos",
                     "Histograma de contagem de reviews recomendados por sentimento", "Gráfico de pizza de distribuição de sentimentos"]

    # Usa o multiselect para definir as opções
    selected_games = st.sidebar.multiselect("Selecione o(s) jogo(s)", game_options)
    selected_reviews = st.sidebar.multiselect("Selecione o tipo de review", list(review_options.keys()))
    selected_graph = st.sidebar.selectbox("Selecione o gráfico", graph_options)

    # Cria um dataframe de dados filtrados baseados nas opções selecionadas
    filtered_data = df[(df["app_name"].isin(selected_games)) & (df["review_score"].isin([review_options[review] for review in selected_reviews]))]

    if selected_graph == "Nuvem de palavras":
        st.subheader("Nuvem de palavras")
        st.write('')
        text = " ".join(review for review in filtered_data.review_text)
        wordcloud = WordCloud(max_words=100, background_color="white").generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt.gcf())
        st.write('Representação em formato de nuvem das palavras mais frequentes')

    elif selected_graph == "Histograma das 10 palavras mais frequentes":
        st.subheader('Histograma das 10 palavras mais frequentes')
        st.write('')
        # Dataframe que contém a contagem de cada palavra nas reviews
        word_counts = filtered_data["review_text"].str.split(expand=True).stack().value_counts().reset_index()
        word_counts.columns = ["word", "count"]
        top_words = word_counts.sort_values("count", ascending=False).head(10)

        plt.figure(figsize=(12, 6))
        ax = top_words.plot(kind="bar", x="word", y="count", rot=45)
        ax.set_xlabel("Palavras", fontsize=16)
        ax.set_ylabel("Contagem", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12, pad=8)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        st.write('Representação gráfica das 10 palavras mais frequentes')

    elif selected_graph == "Histograma de sentimentos":
        st.subheader("Histograma de sentimentos")
        st.write('')

        # Carrega a coluna 'review_score' usando a função carrega_coluna()
        coluna_review_score = carrega_coluna('review_score')

        histograma_sentimentos = go.Figure(data=[
            go.Bar(
                x=['Negativa', 'Positiva'],
                y=coluna_review_score.value_counts().tolist(),
                marker=dict(
                    color=['#FF4136', '#2ECC40'],
                    line=dict(color='#000000', width=1)
                )
            )
        ])

        histograma_sentimentos.update_layout(
            title='Histograma de sentimentos',
            xaxis_title='Polaridade da review',
            yaxis_title='Contagem de registros'
        )

        st.plotly_chart(histograma_sentimentos)
        st.write("Representação gráfica da distribuição de sentimentos em reviews de jogos da Steam")

    elif selected_graph == "Histograma de contagem de reviews recomendados por sentimento":
        st.subheader("Histograma de contagem de reviews recomendados por sentimento")

        # Carregar as colunas relevantes do arquivo Parquet
        df1_recommended = carrega_coluna('review_votes')
        df1_sentiment = carrega_coluna('review_score')

        # Mesclar as colunas relevantes em um único dataframe
        df1 = pd.merge(df1_sentiment, df1_recommended, left_index=True, right_index=True)

        # Renomear os valores das colunas para facilitar a legibilidade
        df1['review_score'] = df1['review_score'].map({-1: 'Negativo', 1: 'Positivo'})
        df1['review_votes'] = df1['review_votes'].map({0: 'Review não recomendada', 1: 'Review recomendada'})

        # Contar a quantidade de reviews recomendadas e não recomendadas para cada sentimento
        sentiment_votes = df1.groupby(['review_score', 'review_votes']).size().unstack('review_votes')

        colors = ['#FF4136', '#2ECC40']

        barras_agrupadas = go.Figure(data=[
            go.Bar(name='Review não recomendada', x=sentiment_votes.index, y=sentiment_votes['Review não recomendada'], 
                   marker=dict(color=colors[0])),
            go.Bar(name='Review recomendada', x=sentiment_votes.index, y=sentiment_votes['Review recomendada'], 
                   marker=dict(color=colors[1]))
        ])

        barras_agrupadas.update_layout(
            title='Contagem de reviews recomendadas e não recomendadas por sentimento',
            xaxis_title='Sentimento',
            yaxis_title='Contagem de registros',
            barmode='stack'
        )

        st.plotly_chart(barras_agrupadas)
        st.write("Representação gráfica da contagem de reviews recomendadas e não recomendadas por sentimento")


    elif selected_graph == "Gráfico de pizza de distribuição de sentimentos":
        st.subheader("Gráfico de pizza de distribuição de sentimentos")

        # Carregar a coluna "review_score" do arquivo Parquet
        df1_sentiment = carrega_coluna('review_score')

        # Contar a quantidade de reviews positivas e negativas
        positivas = df1_sentiment[df1_sentiment['review_score'] == 1].shape[0]
        negativas = df1_sentiment[df1_sentiment['review_score'] == -1].shape[0]

        # Criar o gráfico de pizza com cores ilustrativas
        fig_pizza = go.Figure()

        fig_pizza.add_trace(go.Pie(
            labels=['Negativas', 'Positivas'],
            values=[negativas, positivas],
            marker_colors=['red', 'green'],
            hole=0.3,
        ))

        fig_pizza.update_traces(marker=dict(colors=['red', 'green']))

        fig_pizza.update_layout(
            title="Distribuição de Reviews Positivas e Negativas",
            legend=dict(
                x=1.1,
                y=0.5,
                title="Sentimento",
                title_font=dict(size=14),
                itemsizing='constant'
            )
        )

        st.plotly_chart(fig_pizza)
        st.write("Representação gráfica da distribuição de reviews positivas e negativas")

def main():
    build_header()
    build_body()

if __name__ == "__main__":
    main()






