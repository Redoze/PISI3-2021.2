import streamlit as st
import pandas as pd
import random as rn
import os
import re

@st.cache_resource
def load_csv():
    p = 0.01
    url = "https://www.dropbox.com/s/fy4cffn16usarzk/Dataset_limpo.csv?raw=1"
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

@st.cache_data
def load_csv3(gameid):
    df_time = pd.read_csv("pages/PlayerCountHistory/{}.csv".format(gameid))
    return df_time

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  -  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Lista com os nomes das colunas originais dos dataframes
# Cada coluna foi transormada em um outro dataframe e convertido para parquet
# A coluna principal é a review_text, que precisou ser dividida em partes menores e
# consequentemente precisa ser processada separadamente.
df1_completo = ['app_id', 'app_name', 'review_score', 'review_votes', 'review_text']
df2_completo = ['app_id', 'app_name', 'release_date', 'english', 'developer', 'publisher', 'platforms', 'required_age',
             'categories', 'genres', 'steamspy_tags', 'achievements', 'positive_ratings', 'negative_ratings',
             'average_playtime', 'median_playtime', 'owners', 'price']

path_df1 = 'data/df1/'
path_df2 = 'data/df2/'

# Carrega uma coluna especifica do dataframe
# # # Não passe review_text como parâmetro! # # #
def carrega_df(coluna, path):
    df_coluna = pd.read_parquet(f'{path}/{coluna}.parquet')
    return df_coluna

# Sempre utilize esta função para carregar review_text
def carrega_review_text():
    # Função para extrair o número da parte do nome do arquivo
    def extract_part_number(file_name):
        match = re.search(r'\d+', file_name)
        return int(match.group()) if match else -1

    path_review_text = "data/df1/review_text"

    # Obter a lista de arquivos Parquet na pasta e classificá-los numericamente
    file_list = sorted([file for file in os.listdir(path_review_text) if file.endswith(".parquet")],
                    key=extract_part_number)

    merged_reviews_text = None

    # Ler os data frames e exibi-los com o Streamlit
    for file_name in file_list:
        file_path = os.path.join(path_review_text, file_name)
        df = pd.read_parquet(file_path)

        if merged_reviews_text is None:
            merged_reviews_text = df
        else:
            merged_reviews_text = pd.concat([merged_reviews_text, df], ignore_index=True)

    return merged_reviews_text

# Se for chamar todo o dataframe 1 incluindo o review text, faça primeiro a chamada do review_text!
# Mescla o dataframe a partir da lista com os nomes de suas colunas
# Retorna um dataframe original completo
def mescla_df(df_X_nome_coluna, path):

    # A mesclagem é feita de dois em dois df's, e salva em uma variável.
    # Para mesclar três ou mais é necessário mesclar a variável anterior com o df seguinte
    def primeira_mesclagem(df_X_nome_coluna, path):
        merge_1 = carrega_df(df_X_nome_coluna[0], path)
        return merge_1
    
    # df1_completo possui o review_text
    if df_X_nome_coluna is df1_completo:

        merged_dataframe = primeira_mesclagem(df_X_nome_coluna, path)

        for i in range(1, len(df_X_nome_coluna)):

            if df_X_nome_coluna[i] == 'review_text':
                # Chama o carregamento completo de review_text
                df1_review_text = carrega_review_text()
                merged_dataframe = pd.merge(merged_dataframe, df1_review_text, left_index=True, right_index=True)

            else:
                chama_carregamento = carrega_df(df_X_nome_coluna[i], path)
                merged_dataframe = pd.merge(merged_dataframe, chama_carregamento, left_index=True, right_index=True)

        return merged_dataframe
    
    else:
        merged_dataframe = primeira_mesclagem(df_X_nome_coluna, path)

        for i in range(1, len(df_X_nome_coluna)):
            chama_carregamento = carrega_df(df_X_nome_coluna[i], path)
            merged_dataframe = pd.merge(merged_dataframe, chama_carregamento, left_index=True, right_index=True)

        return merged_dataframe
