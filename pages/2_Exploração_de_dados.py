import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import random as rn
from wordcloud import WordCloud
import plotly.graph_objects as go
from funcs import *

st.set_page_config(
    page_title="Análise de sentimentos em avaliações de jogos na Steam",
    page_icon="✅",
    layout="centered",
)

def build_header():
    st.title("Análise de sentimentos em avaliações de jogos na Steam")
    st.markdown("---")

def build_body():

    ocultar_df = st.sidebar.checkbox('Ocultar conjunto de dados')

    if ocultar_df:
        st.sidebar.write('Conjunto de dados ocultado')

    else:
        st.header('Visão geral do conjunto de dados')
        st.text("")
        pega_df2 = carrega_df('df2')
        pega_df2 = pega_df2.rename(columns={'app_id_df2': 'id', 'app_name_df2': 'nome', 'release_date': 'Lançamento',
                                            'developer': 'Desenvolvedor', 'publisher': 'Publicador', 'platforms': 'Plataforma',
                                            'required_age': 'Faixa etária', 'categories': 'Categorias', 'genres': 'Gêneros',
                                            'steamspy_tags' : 'Tags da Steam', 'achievements': 'Conquistas', 
                                            'positive_ratings': 'Avaliações positivas', 'negative_ratings': 'Avaliações Negativas',
                                            'owners': 'Donos?', 'price': 'Preço'})
        st.dataframe(pega_df2)
        st.caption('A tabela acima apresenta os dados gerais dos jogos utilizados durante o trabalho. Seus dados são provenientes do segundo dataframe.')
        st.text("")

    st.header("Estatísticas gerais dos conjuntos de dados")
    st.text("")

    col1, col2 = st.columns(2)

    with col1:
        carrega_df2_app_id = carrega_coluna('app_id_df2')
        qtd_jogos = len(carrega_df2_app_id) - 1

        carrega_df1_app_id = carrega_coluna('app_id')
        qtd_reviews = len(carrega_df1_app_id) - 1

        qtd_jogos_formatado = '{:,.0f}'.format(qtd_jogos).replace(',', '.')
        qtd_reviews_formatado = '{:,.0f}'.format(qtd_reviews).replace(',', '.')

        st.write('Quantidade total de jogos: %s' % qtd_jogos_formatado)
        st.write('Quantidade total de avaliações: %s' % qtd_reviews_formatado)

    with col2: 
        merged_id_e_review = mistura_colunas('app_id', 'review_text')

        # Calcula a média de reviews por jogo
        media_reviews = merged_id_e_review.groupby('app_id')['app_id'].count().mean()

        media_reviews_formatado = '{:.2f}'.format(round(media_reviews, 2)).replace('.', ',')

        st.write('Quantidade média de avaliações por jogo: %s' % media_reviews_formatado)
    
    
    

build_header()
build_body()
