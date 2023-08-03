import streamlit as st
from funcs import *
from collections import Counter
from streamlit_extras.keyboard_text import key
import plotly.graph_objects as go

st.set_page_config(
    page_title="Análise de sentimentos em avaliações de jogos na Steam",
    page_icon="✅",
    layout="wide",
)

def build_header():

    st.write(f'''<h1 style='text-align: center'
             >Análise de sentimentos em avaliações de jogos na Steam<br><br></h1>
             ''', unsafe_allow_html=True)
    
    st.write(f'''<h2 style='text-align: center; font-size: 18px'>
    Visão geral dos conjuntos de dados, estatísticas gerais dos conjuntos de dados, e avaliação do método de vendas dos jogos nos conjuntos de dados a partir de 3 conjuntos de dados complementares interligados.<br></h2>
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
                 <br>A tabela abaixo apresenta palavras-chave extraídas dos corpo das avaliações dos jogos, assim como o estado da avaliação em si (positiva ou negativa), e o indicador de avaliação recomendada por terceiros. Dados provenientes do primeiro dataframe.<br><br></p>
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
        pega_reviews.rename(columns = {'review_text': 'Avaliações', 'review_score': 'Avaliação do jogo', 
                                       'review_votes': 'Avaliação recomendada'}, inplace=True)
        del pega_reviews['app_id']
        del pega_reviews['app_name']

        ordem_colunas = ['Avaliações', 'Avaliação do jogo', 'Avaliação recomendada']
        # Reordenar as colunas do DataFrame
        pega_reviews = pega_reviews.reindex(columns = ordem_colunas)

    with coluna2_df1:
        st.write('Todas as avaliações de: ', selectbox_jogo_em_app_name_df2)
        st.dataframe(pega_reviews, hide_index=True, width = 800) # Exibe apenas review_text e exclui a coluna de index
    
    with empty2:
        st.empty()

def tabela_dataframe_2():

    st.write(f'''<p style='text-align: center; font-size: 18px'>
                 <br>A tabela abaixo apresenta os dados gerais dos jogos, tais como data de lançamento, desenvolvedora, gênero, etc. Dados provenientes do segundo dataframe.<br><br></p>
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
            <br>A tabela abaixo apresenta dados de contagem de jogadores através do tempo, dentro de um intervalo de aproximadamente 2 a 3 anos (2017-2020), para uma quantidade limitada de jogos. Dados provenientes do terceiro dataframe.<br><br></p>
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
        var = 0

        try:
            pega_df3 = carrega_df(pega_id_do_jogo_procurado)
            pega_df3.rename(columns={'Time': 'Período', 'Playercount': 'Contagem de jogadores'}, inplace=True)
            var += 1

        except FileNotFoundError:
            pass

    with coluna2_df3:

        if var == 0:
            st.warning(f'{selectbox_jogo_em_app_name_df2} não possui referências no conjunto de dados 3, por favor, escolha outro jogo.')

        else:
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

def build_metodo_vendas():
    st.markdown("---")
    st.write(f'''<h2 style='text-align: center'
             >Avaliação do método de vendas</h2>
             ''', unsafe_allow_html=True)
    st.text("")   
    
    empty1, graph_units_sold, empty2 = st.columns([4,10,4])
    
    with empty1:
        st.empty()
        
        
    with graph_units_sold:
    
        units_sold = pd.read_csv("data/vgsales_limpo.csv")
        
        carrega_df1_name = carrega_coluna('app_name')
        
        def calculate_nb_number(reviews, year):
            adjusted_reviews = reviews.copy()
            
            adjusted_reviews.loc[year <2014] *=600
            adjusted_reviews.loc[(year >=2014) & (year<2016)]*=500
            adjusted_reviews.loc[(year >=2016) & (year<2018)]*=400
            adjusted_reviews.loc[(year >=2018) & (year<2020)] *=350
            adjusted_reviews.loc[year >=2020]*=300
            
            return adjusted_reviews
    
        reviews_count = carrega_df1_name.groupby('app_name').size().reset_index(name='reviews')
        
        merge_review_counts = pd.merge(units_sold, reviews_count, left_on='Name', right_on='app_name')
        review_year = merge_review_counts['Year']

        adjusted_reviews_df = calculate_nb_number(merge_review_counts['reviews'], review_year)
        
        merge_review_counts['NB_number'] = adjusted_reviews_df
        #st.write(merge_review_counts)
        

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merge_review_counts['app_name'], y=merge_review_counts['Global_Sales'],
                            mode='lines+markers', name='Unidades Vendidas (em milhões)',
                    line=dict(color='red', width=1), marker=dict(symbol='circle', color ='red', size = 4)))
        fig.add_trace(go.Scatter(x=merge_review_counts['app_name'], y=merge_review_counts['NB_number'],
                            mode='lines+markers', name='Número de Unidades Estimado (Método NB-Number)  (em milhões)',
                    line=dict(color='green', width=1), marker=dict(symbol='circle', color ='green', size = 4)))
        fig.update_layout(
            title='Número Real de Vendas vs Número Estimado',
            width=1250,
            height=800
        )
        st.plotly_chart(fig)
    
    with empty2:
        st.empty()


build_header()
build_body()
build_container()
build_metodo_vendas()
