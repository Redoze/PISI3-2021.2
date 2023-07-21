import streamlit as st
from funcs import *
from collections import Counter

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
                                            'average_playtime': 'Tempo médio de jogo', 'median_playtime': 'Tempo mediano de jogo',
                                            'owners': 'Jogadores únicos', 'price': 'Preço'})

        st.dataframe(pega_df2, hide_index=True)
        st.write('A tabela acima apresenta os dados gerais dos jogos utilizados durante o trabalho. Esses dados são provenientes apenas do segundo dataframe.')
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

        st.write("Categoria mais frequente: ", palavra_mais_frequente)

build_header()
build_body()
