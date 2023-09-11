import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import plotly.graph_objects as go
from funcs import *
import classificadores
import matplotlib.pyplot as plt
from glob import glob

st.set_page_config(
    page_title="Simulação do seu jogo",
    page_icon="🔎",
    layout="centered",
)

def build_header():
    st.write(f'''<h1 style='text-align: center'
             >Ferramenta de Simulação de Desempenho de Jogo<br></h1>
             ''', unsafe_allow_html=True)

    st.write(f'''<p style='text-align: center'>
            Bem-vindo à nossa ferramenta interativa que permite simular o jogo de acordo com diversas características. Explore as opções abaixo para personalizar sua simulação.</p>
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
        
        selected_games2 = filtered_data['app_id_df2'].unique()
        func_names = ["keyword_extraction_and_word_cloud", "player_count_and_units_sold_graph"]

        for funcoes in func_names:
            chama_funcao = globals()[func_names[func_names.index(funcoes)]]
            chama_funcao(filtered_data, df_reviews, selected_games2)

    #chamando o modelo de machine learning e a word cloud
    #keyword_extraction_and_word_cloud(filtered_data)
    #player_count_and_units_sold_graph(filtered_data)

    ############################################################ - ############################################################

    vazio_1, coluna_1, coluna_2, vazio_2 = st.columns([1,5,5,1])

    with vazio_1:
        st.empty()
            
    with coluna_1:
        # Usa o multiselect para definir as opções
        selected_genre = st.multiselect("Selecione o(s) gênero(s)", genre_options)
        
    with coluna_2:
        selected_platform = st.multiselect("Selecione uma plataforma para o jogo:", platform_options)
    
    with vazio_2:
        st.empty()

    ############################################################ - ############################################################

    vazio_1_lv_2, coluna_1_lv_2, coluna_2_lv_2, vazio_2_lv_2 = st.columns([1,5,5,1])

    with vazio_1_lv_2:
        st.empty()
            
    with coluna_1_lv_2:
        # Usa o multiselect para definir as opções
        selected_price = st.multiselect("Selecione se o jogo é gratuito ou não:", list(price_options))
        
    with coluna_2_lv_2:
        selected_category = st.multiselect("Selecione a(s) categoria(s) do jogo:", category_options)
    
    with vazio_2_lv_2:
        st.empty()

    
    inicia_simulacao(0)

def keyword_extraction_and_word_cloud(filtered_data, df_reviews, variavel_gambiarra):
    #extrai os app_ids dos jogos baseados nos critérios de seleção
    selected_games = filtered_data['app_id_df2'].unique()
    
    try:
        #filtra os dados para selecionar apenas as reviews dos jogos selecionados
        filtered_reviews = classificadores.naive(df_reviews[df_reviews["app_id"].isin(selected_games)])[6]
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
        positive_cloud = WordCloud(width=1000, height=600, background_color="black", max_words=25).generate_from_frequencies(positive_freq_dict)
        negative_cloud = WordCloud(width=1000, height=600, background_color="black", max_words=25).generate_from_frequencies(negative_freq_dict)

        st.write(f'''<p style='text-align: center'>
                 Nuvem de palavras de avaliações positivas</p>
                ''', unsafe_allow_html=True)
        plt.imshow(positive_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.image(positive_cloud.to_image())
        
        st.write(f'''<p style='text-align: center'>
                 Nuvem de palavras de avaliações negativas</p>
                ''', unsafe_allow_html=True)
        
        plt.imshow(negative_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.image(negative_cloud.to_image())
    except ValueError:
        st.write(f'''<p style='text-align: center'>
                <br>Selecione todos os atributos primeiro / Dados insuficientes para esses parâmetros.</p>
                ''', unsafe_allow_html=True)
    pass

def player_count_and_units_sold_graph(df, variavel_gambiarra2, selected_games):
    dfs = []
    for game in selected_games:
        try:
            df_pc = carrega_df(game)
            df_pc['Time'] = pd.to_datetime(df_pc['Time'])

            #unindo o df de playercount com o df filtrado
            df_pc = pd.merge(df_pc, df, how='inner', left_on=df_pc.index, right_on='app_id_df2')
            df_pc.set_index('Time', inplace=True)

            #computando as vendas estimadas
            df_pc['estimated_sales'] = estimate_sales(df_pc['release_date'], df_pc['Playercount'])

            #computando a media de jogadores diarios com as vendas estimadas
            df_daily = df_pc.resample('D').agg({'Playercount': 'mean', 'estimated_sales': 'mean'})

            dfs.append(df_daily)
        except FileNotFoundError:
            pass
    if not dfs:
        st.write(f'''<p style='text-align: center'>
                Sem dados para exibir.</p>
                ''', unsafe_allow_html=True)
        st.stop()
    playercount_df = pd.concat(dfs)
    
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(playercount_df.index, playercount_df['estimated_sales'], color='tab:blue')
    ax1.set_ylabel('Vendas estimadas', color='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(playercount_df.index, playercount_df['Playercount'], color='tab:red')
    ax2.set_ylabel('Quantidade de jogadores', color='tab:red')

    plt.title('Média de vendas e contagem de jogadores ao longo do tempo')
    st.pyplot(fig)
    pass

def main():
    build_header()
    build_body()

if __name__ == "__main__":
    main()
