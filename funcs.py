import streamlit as st
import pandas as pd
import random as rn
import os
import re
import numpy as np
from scipy import stats

# Versão antiga de leitura dos dataframes. Ignorem e passem a usar a nova versão mais abaixo.
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

# Os dataframes foram divididos pelas suas colunas, agora cada coluna passou a ser um arquivo independente que também foi
# convertido para o formato parquet.
# Os principais dados do projeto estão contidos na coluna "review_text" do dataframe 1. Pelo seu tamanho ser muito elevado,
# foi necessário dividi-lo em partes menores. Ou seja, ele possui uma função própria para processá-lo.

# Dicionário com os dataframes e suas colunas.
dataframes = {'df1':['app_id', 'app_name', 'review_score', 'review_votes', 'review_text'],
              'df2':['app_id_df2', 'app_name_df2', 'release_date', 'english', 'developer', 'publisher', 'platforms', 'required_age',
             'categories', 'genres', 'steamspy_tags', 'achievements', 'positive_ratings', 'negative_ratings',
             'average_playtime', 'median_playtime', 'owners', 'price']}

# Retorna o caminho do dataframe da coluna escolhida
@st.cache_resource
def procura_coluna(nome_coluna):

    path_df1 = 'data/df1/'
    path_df2 = 'data/df2/'

    for df, colunas in dataframes.items():
        if nome_coluna in colunas:
            if df == 'df1':
                return path_df1
            if df == 'df2':
                return path_df2
    return None

# Carrega uma coluna especifica
# # # Não passe review_text como argumento! # # #
@st.cache_resource
def carrega_coluna(coluna):
    path = procura_coluna(coluna)
    df_coluna = pd.read_parquet(f'{path}/{coluna}.parquet')
    return df_coluna

# Sempre utilize esta função para carregar review_text
@st.cache_resource
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

# Fundi duas colunas. Os parâmetros são: nome da primeira coluna e o caminho de seu dataframe e o nome da segunda coluna
# e seu caminho de seu dataframe.
# Esta função também pode misturar colunas de diferentes dataframes, por isso cada coluna recebe seu devido caminho.
@st.cache_resource
def mistura_colunas(coluna1, coluna2):

    if coluna1 == 'review_text' or coluna2 == 'review_text':
        if coluna1 == 'review_text':
                    coluna1 = carrega_review_text()
                    coluna2 = carrega_coluna(coluna2)
                    merge = pd.merge(coluna1, coluna2, left_index=True, right_index=True)
                    return merge
        
        else:
            coluna1 = carrega_coluna(coluna1)
            coluna2 = carrega_review_text()
            merge = pd.merge(coluna1, coluna2, left_index=True, right_index=True)
            return merge

    else:
        coluna1 = carrega_coluna(coluna1)
        coluna2 = carrega_coluna(coluna2)
        merge = pd.merge(coluna1, coluna2, left_index=True, right_index=True)

        return merge

# Retorna o dataframe completo
@st.cache_resource
def carrega_df(nome_df):
    
    if nome_df == 'df1':
        df1_colunas = dataframes['df1']

        # Para carregar um dataframe completo ou duas ou mais colunas, é primeiramente preciso fundir as duas primeiras colunas.
        # Para só depois fundir as duas primeiras juntas com a seguinte. Motivo que faz com que o for abaixo comece na posição 1.
        merged_dataframe = carrega_coluna(df1_colunas[0])

        for i in range(1, len(df1_colunas)):

            if df1_colunas[i] != 'review_text':
                chama_carregamento = carrega_coluna(df1_colunas[i])
                merged_dataframe = pd.merge(merged_dataframe, chama_carregamento, left_index=True, right_index=True)
            elif df1_colunas[i] == 'review_text':
                chama_carregamento = carrega_review_text()
                merged_dataframe = pd.merge(merged_dataframe, chama_carregamento, left_index=True, right_index=True)
                
        return merged_dataframe

    if nome_df == 'df2':
        df2_colunas = dataframes['df2']
        # Mesmo propósito do uso acima.
        merged_dataframe = carrega_coluna(df2_colunas[0])

        for i in range(1, len(df2_colunas)):
            chama_carregamento = carrega_coluna(df2_colunas[i])
            merged_dataframe = pd.merge(merged_dataframe, chama_carregamento, left_index=True, right_index=True)

        return merged_dataframe

    else:
        file_path = os.path.join("data/df3/", nome_df)
        df3 = pd.read_parquet(f'{file_path}/{nome_df}.parquet')
        return df3
    
    
def remove_outliers_zscore(df, columns, threshold=3):
    z_scores = np.abs(stats.zscore(df[columns]))
    return df[(z_scores < threshold).all(axis=1)]
