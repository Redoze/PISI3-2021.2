import streamlit as st
from funcs import *
from collections import Counter
from streamlit_extras.keyboard_text import key

st.set_page_config(
    page_title="Análise de sentimentos em avaliações de jogos na Steam",
    page_icon="✅",
    layout="wide",
)

def build_header():

    st.write(f'''<h1 style='text-align: center'
             >Análise de sentimentos em avaliações de jogos na Steam<br><br></h1>
             ''', unsafe_allow_html=True)
    
    st.write(f'''<h2 style='text-align: center; font-size: 20px'>
    PLACEHOLDERPLACEHOLDERPLACEHOLDERPLACEHOLDERPLACEHOLDER<br></h2>
        ''', unsafe_allow_html=True)
    st.markdown("---")

def build_body():

    st.write(f'''<h2 style='text-align: center; font-size: 36px'>
            Visão geral do conjunto de dados</h2>
             ''', unsafe_allow_html=True) # 36px equivalem ao h2/subheader
    st.text("")

    font_css = """
    <style>
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 18px; } </style>""" # Define o tamanho da fonte do texto das tabelas.

    st.write(font_css, unsafe_allow_html=True)
    
    def center_text_with_em_spaces(text, whitespace):
        return text.center(whitespace, "\u2001")

    # Defina a lista de rótulos das abas
    listTabs = ["Conjunto de dados 1", "Conjunto de dados 2", "Conjunto de dados 3"]

    # Defina o tamanho desejado para preenchimento e centralização
    whitespace = 48

    # Centraliza os rótulos das abas sem usar um loop for
    centered_tabs = [center_text_with_em_spaces(s, whitespace) for s in listTabs]

    # Cria as tabelas para exibição dos dataframes
    dataframe_1, dataframe_2, dataframe_3 = st.tabs(centered_tabs)

    with dataframe_1:
        tabela_dataframe_1()

    with dataframe_2:
        tabela_dataframe_2()
        
    with dataframe_3:
        tabela_dataframe_3()

def tabela_dataframe_1():

    st.write(f'''<p style='text-align: center; font-size: 18px'>
                 <br>PLACEHOLDERPLACEHOLDERPLACEHOLDERPLACEHOLDERPLACEHOLDER<br><br></p>
                 ''', unsafe_allow_html=True)

    empty1, coluna1_df1, coluna2_df1, empty2 = st.columns([2,2,6,1], gap="large")

    with empty1:
        st.empty()

    with coluna1_df1:

        # Também poderia estar chamando o app_name do df1 e passar '(...)['app_name'].unique()' para obter apenas os valores únicos dela
        # não a usei por o app_name do df2 ser suficiente e ainda ser muito mais leve.
        chama_name_df2 = carrega_coluna('app_name_df2')
        selectbox_jogo_em_app_name_df2 = st.selectbox("Selecione um jogo:", chama_name_df2) # Seleciona o jogo pelo nome

        chama_id_name_df2 = mistura_colunas('app_id_df2','app_name_df2')
        # Pega o id do jogo pelo nome do jogo selecionado anteriormente
        pega_id_do_jogo_procurado = chama_id_name_df2[chama_id_name_df2['app_name_df2'] == selectbox_jogo_em_app_name_df2]['app_id_df2'].iloc[0]

        chama_df1 = carrega_df('df1')
        # Encontra todas as linhas de chama_df1 com o mesmo valor de id encontrado acima
        pega_reviews = chama_df1.loc[chama_df1['app_id'] == pega_id_do_jogo_procurado]
        # Renomeia a coluna, precisou do inplace=True para aplicar a renoeação diretamente no df a menos que seja feita uma nova variavel para tal
        pega_reviews.rename(columns = {'review_text': 'Avaliações', }, inplace=True)
        del pega_reviews['app_id']
        del pega_reviews['app_name']

        ordem_colunas = ['Avaliações', 'review_score', 'review_votes']
        # Reordenar as colunas do DataFrame
        pega_reviews = pega_reviews.reindex(columns = ordem_colunas)

    with coluna2_df1:
        st.write('Todas as valiações de: ', selectbox_jogo_em_app_name_df2)
        st.dataframe(pega_reviews, hide_index=True, width = 800) # Exibe apenas review_text e exclui a coluna de index
    
    with empty2:
        st.empty()

def tabela_dataframe_2():

    st.write(f'''<p style='text-align: center; font-size: 18px'>
                 <br>A tabela abaixo apresenta os dados gerais dos jogos utilizados durante o trabalho. Esses dados são provenientes apenas do segundo dataframe.<br><br></p>
                 ''', unsafe_allow_html=True)

    pega_df2 = carrega_df('df2')
    pega_df2.rename(columns={'app_id_df2': 'id', 'app_name_df2': 'Nome', 'release_date': 'Lançamento', 'english': 'Inglês', 
                                'developer': 'Desenvolvedor', 'publisher': 'Publicador', 'platforms': 'Plataforma',
                                'required_age': 'Faixa etária', 'categories': 'Categorias', 'genres': 'Gêneros',
                                'steamspy_tags' : 'Tags da Steam', 'achievements': 'Conquistas', 
                                'positive_ratings': 'Avaliações positivas', 'negative_ratings': 'Avaliações Negativas',
                                'average_playtime': 'Tempo médio de jogo', 'median_playtime': 'Tempo mediano de jogo',
                                'owners': 'Jogadores únicos', 'price': 'Preço'}, inplace=True)

    st.dataframe(pega_df2, hide_index=True)

def tabela_dataframe_3():

    st.write(f'''<p style='text-align: center; font-size: 18px'>
            <br>PLACEHOLDERPLACEHOLDERPLACEHOLDERPLACEHOLDERPLACEHOLDER<br><br></p>
            ''', unsafe_allow_html=True)

    empty1, coluna1_df3, coluna2_df3, empty2 = st.columns([5,2,6,1], gap="large")

    with empty1:
        st.empty()

    with coluna1_df3:

        # Também poderia estar chamando o app_name do df1 e passar '(...)['app_name'].unique()' para obter apenas os valores únicos dela
        # não a usei por o app_name do df2 ser suficiente e ainda ser muito mais leve.
        chama_name_df2 = carrega_coluna('app_name_df2')
        selectbox_jogo_em_app_name_df2 = st.selectbox("Escolha um jogo:", chama_name_df2) # Seleciona o jogo pelo nome

        chama_id_name_df2 = mistura_colunas('app_id_df2','app_name_df2')
        # Pega o id do jogo pelo nome do jogo selecionado anteriormente
        pega_id_do_jogo_procurado = chama_id_name_df2[chama_id_name_df2['app_name_df2'] == selectbox_jogo_em_app_name_df2]['app_id_df2'].iloc[0]

        pega_df3 = carrega_df(pega_id_do_jogo_procurado)
        pega_df3.rename(columns={'Time': 'Período', 'Playercount': 'Contagem de jogadores'}, inplace=True)

    with coluna2_df3:
        st.write(selectbox_jogo_em_app_name_df2)
        st.dataframe(pega_df3, hide_index=True) # Exibe apenas review_text e exclui a coluna de index
    
    with empty2:
        st.empty()

def build_container():

    st.markdown("---")
    st.write(f'''<h2 style='text-align: center'
             >Estatísticas gerais dos conjuntos de dados</h2>
             ''', unsafe_allow_html=True)
    st.text("")

    empty1, coluna1, coluna2, empty2 = st.columns([4,2,2,4])

    with empty1:
        st.empty()

    with coluna1:

        carrega_df2_app_id = carrega_coluna('app_id_df2')
        qtd_jogos = len(carrega_df2_app_id) - 1

        carrega_df1_app_id = carrega_coluna('app_id')
        qtd_reviews = len(carrega_df1_app_id) - 1

        qtd_jogos_formatado = '{:,.0f}'.format(qtd_jogos).replace(',', '.')
        qtd_reviews_formatado = '{:,.0f}'.format(qtd_reviews).replace(',', '.')

        st.write('Quantidade total de jogos: %s' % qtd_jogos_formatado)
        st.write('Quantidade total de avaliações: %s' % qtd_reviews_formatado)

    with coluna2: 

        merged_id_e_review = mistura_colunas('app_id', 'review_text')

        # Calcula a média de reviews por jogo
        media_reviews = merged_id_e_review.groupby('app_id')['app_id'].count().mean()

        media_reviews_formatado = '{:.2f}'.format(round(media_reviews, 2)).replace('.', ',')

        st.write('Quantidade média de avaliações por jogo: %s' % media_reviews_formatado)

        # Função para encontrar a palavra mais comum no DataFrame
        def palavra_mais_comum(dataframe):
            # Transforma a coluna em uma lista de palavras separadas por ";"
            words_list = ';'.join(dataframe['categories']).split(';')

            # Conta a frequência de cada palavra usando Counter
            word_count = Counter(words_list)

            # Encontra a palavra mais comum
            most_common_word = word_count.most_common(1)[0][0]

            return most_common_word

        # Lê o DataFrame usando a função chama_coluna
        df = carrega_coluna('categories')

        # Encontra a palavra mais comum em todo o DataFrame
        palavra_mais_frequente = palavra_mais_comum(df)

        st.write("Categoria mais presente: ", palavra_mais_frequente)

    with empty2:
        st.empty()

build_header()
build_body()
build_container()
