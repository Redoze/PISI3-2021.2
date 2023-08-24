import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import plotly.graph_objects as go
from funcs import *
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Simulação do seu jogo",
    page_icon="🔎",
    layout="wide",
)

def build_header():
    st.write(f'''<h1 style='text-align: center'
             >Ferramenta de simulação de um jogo<br><br></h1>
             ''', unsafe_allow_html=True)
    st.markdown("---")

def build_body():
    #carregando dados
    df = carrega_df('df2')
    df_reviews = carrega_df('df1')
    df_reviews["sentiment"] = df_reviews["review_score"].apply(lambda x: 1 if x == 1 else 0)

    #função para tratar as categorias unidas numa string com ";"
    def get_unique_options(df, column):
        options_set = set()
        for row in df[column]:
            split_values = row.split(';')
            options_set.update(split_values)
        return list(options_set)

    
    #definindo as colunas
    genre_options = get_unique_options(df, "genres")
    platform_options = get_unique_options(df, "platforms")
    category_options = get_unique_options(df, "categories")
    price_options = df['price'].apply(lambda x: 'Free' if x == float(0) else 'Paid').unique()

    #selected_genre = st.multiselect("Selecione um gênero para o jogo:", genre_options)
    #selected_platform = st.multiselect("Selecione uma plataforma para o jogo:", platform_options)
    #selected_price = st.multiselect("Selecione um se o jogo é gratuito ou não:", list(price_options.keys()))
    #selected_category = st.multiselect("Selecione a categoria do jogo:", category_options)

    

    def inicia_simulacao(posicao):
        #filtrando os dados
        filtered_data = df[(df["genres"].isin(selected_genre)) &
                           (df["platforms"].isin(selected_platform)) &
                           (df["price"].apply(lambda x: 'Free' if x == float(0) else 'Paid').isin(selected_price)) &
                           (df["categories"].isin(selected_category))]
        
        chama_funcao = globals()["keyword_extraction_and_word_cloud"]
        chama_funcao(filtered_data, df_reviews)

    #chamando o modelo de machine learning e a word cloud
    #keyword_extraction_and_word_cloud(filtered_data)
    #player_count_and_units_sold_graph(filtered_data)

    ############################################################ - ############################################################

    vazio_1_lv_2, coluna_1_lv_2, coluna_2_lv_2, vazio_2_lv_2 = st.columns([3,2,2,3])

    with vazio_1_lv_2:
        st.empty()
            
    with coluna_1_lv_2:
        # Usa o multiselect para definir as opções
        selected_genre = st.multiselect("Selecione o(s) gênero(s)", genre_options)
        
    with coluna_2_lv_2:
        selected_platform = st.multiselect("Selecione uma plataforma para o jogo:", platform_options)
    
    with vazio_2_lv_2:
        st.empty()

    ############################################################ - ############################################################

    vazio_1_2v_2, coluna_1_2v_2, coluna_2_2v_2, vazio_2_2v_2 = st.columns([3,2,2,3])

    with vazio_1_2v_2:
        st.empty()
            
    with coluna_1_2v_2:
        # Usa o multiselect para definir as opções
        selected_price = st.multiselect("Selecione um se o jogo é gratuito ou não:", list(price_options))
        
    with coluna_2_2v_2:
        selected_category = st.multiselect("Selecione a(s) categoria(s) do jogo:", category_options)
    
    with vazio_2_2v_2:
        st.empty()

    ############################################################ Exibição dos dados ############################################################

    vazio_1_lv_3, coluna_1_lv_3, vazio_2_lv_3 = st.columns([1,5,1])
    
    with vazio_1_lv_3:
        st.empty()

    with coluna_1_lv_3:
        inicia_simulacao(0)
            
    with vazio_2_lv_3:
        st.empty()   


def keyword_extraction_and_word_cloud(filtered_data, df_reviews):
    
    #extrai os app_ids dos jogos baseados nos critérios de seleção
    selected_games = filtered_data['app_id_df2'].unique()

    #filtra os dados para selecionar apenas as reviews dos jogos selecionados
    filtered_reviews = df_reviews[df_reviews["app_id"].isin(selected_games)]
    
    try:
        #separando reviews positivas das negativas
        positive_reviews = filtered_reviews[filtered_reviews["sentiment"]==1]
        negative_reviews = filtered_reviews[filtered_reviews["sentiment"]==0]

        #extrai keywords com o modelo tf-idf para reviews positivas e negativas
        vectorizer = TfidfVectorizer()
        
        positive_keywords = vectorizer.fit_transform(positive_reviews['review_text'])
        positive_features = vectorizer.get_feature_names_out()
        positive_freqs = np.array(positive_keywords.sum(axis=0))[0]
        positive_freq_dict = dict(zip(positive_features, positive_freqs))

        negative_keywords = vectorizer.fit_transform(negative_reviews['review_text'])
        negative_features = vectorizer.get_feature_names_out()
        negative_freqs = np.array(negative_keywords.sum(axis=0))[0]
        negative_freq_dict = dict(zip(negative_features, negative_freqs))

        #gerando as wordclouds
        positive_cloud = WordCloud(width=800, height=400, background_color="black", max_words=25).generate_from_frequencies(positive_freq_dict)
        negative_cloud = WordCloud(width=800, height=400, background_color="black", max_words=25).generate_from_frequencies(negative_freq_dict)

        st.write("Word Cloud para reviews positivas:", unsafe_allow_html=True)
        plt.imshow(positive_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.image(positive_cloud.to_image())
        
        st.write("Word Cloud para reviews negativas:", unsafe_allow_html=True)
        plt.imshow(negative_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.image(negative_cloud.to_image())
    except ValueError:
        st.write("Selecione todos os atributos primeiro.")
    pass

def player_count_and_units_sold_graph(df):
    # Your graphing function here...
    pass

def main():
    build_header()
    build_body()

if __name__ == "__main__":
    main()
