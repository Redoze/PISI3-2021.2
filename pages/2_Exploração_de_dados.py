import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
import plotly.graph_objects as go
from funcs import *
import matplotlib.pyplot as plt

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Exploração de dados",
    page_icon="🔎",
    layout="centered",
)

def build_header():
    st.write(f'''<h1 style='text-align: center'>
             Exploração de dados<br><br></h1>''', unsafe_allow_html=True)
    
    st.write(f'''<p style='text-align: center'>
             Investigação e análise dos conjuntos de dados - Visualização dos dados, detecção de padrões, identificação de outliers, etc.<br></p>''', unsafe_allow_html=True)
    st.markdown("---")

def build_body():
    st.write(f'''<h2 style='text-align: center; font-size: 28px'>
            Utilize os filtros para explorar os dados<br></h2>''', unsafe_allow_html=True) # 28px aprox. tamanho do subheader
        
    # Carrega os dataframes
    df = carrega_df('df1')

    # Define os itens a serem selecionados na lista dropdown
    game_options = df["app_name"].dropna().unique() # Adicionado o método 'dropna()' para remover os valores nulos.
    review_options = {"Negativa": -1, "Positiva": 1}

    graph_options = {"Nuvem de palavras": ['grafico_1', 'game/review', 'filtered_data'],
                     "Histograma de sentimentos": ['grafico_2', False],
                     "Histograma de contagem de reviews recomendados por sentimento": ['grafico_3', False], 
                     "Relação entre avaliações e tempo de jogo": ['grafico_4', False], 
                     "Correlação entre a polaridade média das reviews e a quantidade média de jogadores": ['grafico_5', 'game/review', 'df'], 
                     "Correlação entre a quantidade média de jogadores e quantidade média de reviews indicadas como úteis": ['grafico_6', 'game/review', 'df']}
    # A string da posição 1 da lista dos valores das chaves representa uma condicional para mostrar ou não as demais caixas de seleção.
    # False         - sem instrução
    # 'game/review' - precisa de um jogo e avaliação
    # A string da posição 2 da lista dos valores das chaves representa o argumento extra que o gráfico precisa. Podendo passar o argumento extra 'df' ou 'filtered_data'.

    def inicia_grafico():
        
        if graph_options[selected_graph][1] == 'game/review':
            # Cria um dataframe de dados filtrados baseados nas opções selecionadas
            filtered_data = df[(df["app_name"].isin(selected_games)) & (df["review_score"].isin([review_options[review] for review in selected_reviews]))]

        for nome_funcao, graficos in graph_options.items():
            if nome_funcao == selected_graph:
                chama_funcao = globals()[graficos[0]]

                if graph_options[selected_graph][1] == False:
                    chama_funcao()

                elif graph_options[selected_graph][1] == 'game/review' and graph_options[selected_graph][2] == 'df':
                    chama_funcao(selected_games, selected_reviews, df)

                elif graph_options[selected_graph][1] == 'game/review' and graph_options[selected_graph][2] == 'filtered_data':
                    chama_funcao(selected_games, selected_reviews, filtered_data)

    ############################################################ SELEÇÃO DO GRÁFICO ############################################################

    vazio_1, coluna_1, vazio_2 = st.columns([2,5,2])

    with vazio_1:
        st.empty()

    with coluna_1:
        selected_graph = st.selectbox("Selecione um gráfico: ", graph_options)

    with vazio_2:
        st.empty()
    
    ############################################################ CONDICIONAIS DE JOGO E AVALIÇÃO ############################################################

    if graph_options[selected_graph][1] == 'game':   # Se o gráfico precisa apenas do input de jogo.

        vazio_1, coluna_1, vazio_2 = st.columns([1,3,1])

        with vazio_1:
            st.empty()

        with coluna_1:
            selected_games = st.multiselect("Selecione o(s) jogo(s)", game_options)
        
        with vazio_2:
            st.empty()

        inicia_grafico()
        
    elif graph_options[selected_graph][1] == 'review':   # Se o gráfico precisa apenas do input de avalição.

        selected_reviews = st.multiselect("Selecione o tipo de avaliação", list(review_options.keys()))
        inicia_grafico()

    elif graph_options[selected_graph][1] == 'game/review':   # Se o gráfico precisa do input de jogo e avalição.

        vazio_1, coluna_1, coluna_2, vazio_2 = st.columns([1,3,3,1])

        with vazio_1:
            st.empty()
                
        with coluna_1:
            selected_games = st.multiselect("Selecione o(s) jogo(s)", game_options)
            
        with coluna_2:
            selected_reviews = st.multiselect("Selecione o tipo de avaliação", list(review_options.keys()))
        
        with vazio_2:
            st.empty()

        inicia_grafico()

    else: # Se o gráfico não precisa de nenhum input.
        inicia_grafico()
    
    ############################################################ SELEÇÃO STRING ############################################################

def compara_selecao(plural, selected_games, selected_reviews):
    # plural: recebe False ou True e diz se o gráfico lida com mais de uma entrada de jogo/avaliação
    # Se o retorno da variável caso for igual a 0, nenhuma mensangem é exibida/retornada e o gráfico funciona normalmente.

    caso = 0
    texto = ""

    texto_1 = "Por favor, selecione ao menos um jogo e um tipo de avaliação."
    texto_2 = "Por favor, selecione ao menos um tipo de avaliação."
    texto_3 = "Por favor, selecione ao menos um jogo."

    texto_1p = "Por favor, selecione alguns jogos e ao menos um tipo de avaliação."
    texto_3p = "Por favor, selecione alguns jogos."

    if len(selected_games) == 0 and len(selected_reviews) == 0:
        caso = 1

        if plural == True:
            texto = texto_1p
        else:
            texto = texto_1

    elif len(selected_games) != 0 and len(selected_reviews) == 0:
        caso = 2
        texto = texto_2

    elif len(selected_games) == 0 and len(selected_reviews) != 0:
        caso = 3

        if plural == True:
            texto = texto_3p
        else:
            texto = texto_3

    return caso, texto

######################################################### GRÁFICOS ########################################################

def grafico_1(selected_games, selected_reviews, filtered_data):
    var_compara_selecao = compara_selecao(False, selected_games, selected_reviews)

    if var_compara_selecao[0] == 0:
        text = " ".join(review for review in filtered_data.review_text)

        wordcloud = WordCloud(max_words=100, background_color="black").generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.legend().set_visible(False)

        st.write(f'''<h3 style='text-align: center'><br>
                Nuvem de palavras<br><br></h3>
                ''', unsafe_allow_html=True)
        
        # Converte o gráfico do Matplotlib em imagem e use st.image()
        image = wordcloud.to_image()
        st.image(image, use_column_width=True)
        
        st.write(f'''<p style='text-align: center'>
                <br>Nuvem de Palavras para os jogos selecionados, mostrando a partir dos dados brutos algumas keywords que se destacam entre as reviews.</p>
                ''', unsafe_allow_html=True)
        
    else:
        st.write(f'''<p style='text-align: center'>
                <br>{var_compara_selecao[1]}</p>
                ''', unsafe_allow_html=True)

def grafico_2():

    st.write(f'''<h3 style='text-align: center'>
             <br>Histograma de sentimentos<br><br></h3>
            ''', unsafe_allow_html=True)

    # Carrega a coluna 'review_score' usando a função carrega_coluna()
    coluna_review_score = carrega_coluna('review_score')

    histograma_sentimentos = go.Figure(data=[
        go.Bar(
            x=['Positiva', 'Negativa'],
            y=coluna_review_score.value_counts().tolist(),
            marker=dict(
                color=['#2ECC40', '#FF4136'],
                line=dict(color='#000000', width=1)
            )
        )
    ])

    histograma_sentimentos.update_layout(
        title='',
        xaxis_title='Polaridade da avaliação',
        yaxis_title='Contagem de registros'
    )

    st.plotly_chart(histograma_sentimentos)
    st.write(f'''<p style='text-align: center'>
             Representação gráfica da distribuição de sentimentos nas avaliações dos jogos da Steam</p>
             ''', unsafe_allow_html=True)

def grafico_3():

    st.write(f'''<h3 style='text-align: center'><br>
        Histograma de contagem de avaliações recomendadas por sentimento<br><br></h3>
            ''', unsafe_allow_html=True)
    
    # Carregar as colunas relevantes do arquivo Parquet
    df1_recommended = carrega_coluna('review_votes')
    df1_sentiment = carrega_coluna('review_score')

    # Mesclar as colunas relevantes em um único dataframe
    df1 = pd.merge(df1_sentiment, df1_recommended, left_index=True, right_index=True)

    # Renomear os valores das colunas para facilitar a legibilidade
    df1['review_score'] = df1['review_score'].map({-1: 'Negativo', 1: 'Positivo'})
    df1['review_votes'] = df1['review_votes'].map({0: 'Avaliação não recomendada', 1: 'Avaliação recomendada'})

    # Contar a quantidade de reviews recomendadas e não recomendadas para cada sentimento
    sentiment_votes = df1.groupby(['review_score', 'review_votes']).size().unstack('review_votes')

    colors = ['#FF4136', '#2ECC40']

    barras_agrupadas = go.Figure(data=[
        go.Bar(name='Avaliação não recomendada', x=sentiment_votes.index, y=sentiment_votes['Avaliação não recomendada'], 
                marker=dict(color=colors[0])),
        go.Bar(name='Avaliação recomendada', x=sentiment_votes.index, y=sentiment_votes['Avaliação recomendada'], 
                marker=dict(color=colors[1]))
    ])

    barras_agrupadas.update_layout(
        title='',
        xaxis_title='Sentimento',
        yaxis_title='Contagem de registros',
        barmode='stack'
    )

    st.plotly_chart(barras_agrupadas)
    st.write(f'''<p style='text-align: center'>
             Representação gráfica da contagem de avaliações recomendadas e não recomendadas por sentimento</p>
             ''', unsafe_allow_html=True)

def grafico_4():

    st.write(f'''<h3 style='text-align: center'><br>
        Relação entre avaliações e tempo de jogo<br><br></h3>
            ''', unsafe_allow_html=True)

    # Carrega apenas as colunas necessárias do DataFrame
    columns_to_load = ['app_name_df2', 'genres', 'average_playtime', 'positive_ratings', 'negative_ratings']
    df_columns = [carrega_coluna(col) for col in columns_to_load]

    # Criando uma barra deslizante (slider) para o threshold
    threshold = st.slider("Selecione o valor para aumentar ou diminiur a quantidade de outliers", min_value=1, max_value=30, value=3, step=1)
    
    # Definindo as colunas relevantes para o gráfico
    columns_for_graph = ['average_playtime', 'positive_ratings', 'negative_ratings']

    # Restante do código para detecção e remoção de outliers
    filtered_data = pd.concat(df_columns, axis=1)

    # Remove outliers usando Z-score para as colunas relevantes
    filtered_data = remove_outliers_zscore(filtered_data, columns_for_graph,threshold=threshold)

        # Criando um botão para alternar entre mostrar e ocultar os outliers
    show_outliers = st.checkbox("Desmarque para mostrar o gráfico sem Outliers", value=True)

    if show_outliers:
        # Mostrar o gráfico com outliers
        data_to_plot = pd.concat(df_columns, axis=1)
    else:
        # Mostrar o gráfico sem outliers
        data_to_plot = filtered_data

    colors = ['#FF4136', '#2ECC40']
    sizes=10

    #Cria um scatter com cores diferentes para cada categoria de avaliação(Positiva, Negativa)
    fig = go.Figure()
        # Adicionando o scatter plot para as avaliações positivas
    fig.add_trace(
        go.Scatter(
            x=data_to_plot['average_playtime'],
            y=data_to_plot['positive_ratings'],
            mode='markers',
            name='Avaliações Positivas',
            marker=dict(color=colors[1], size=sizes),
            text=[f'Jogo: {name}<br>Gênero: {genre}<br>Tempo Médio de Jogo: {playtime}<br>Avaliações Positivas: {pos_ratings}'
                for name, genre, playtime, pos_ratings in
                zip(data_to_plot['app_name_df2'], data_to_plot['genres'], data_to_plot['average_playtime'], data_to_plot['positive_ratings'])],
            hovertemplate='%{text}<extra></extra>'
        ))

    # Adicionando o scatter plot para as avaliações negativas
    fig.add_trace(
        go.Scatter(
            x=data_to_plot['average_playtime'],
            y=data_to_plot['negative_ratings'],
            mode='markers',
            name='Avaliações Negativas',
            marker=dict(color=colors[0], size=sizes),
            text=[f'Jogo: {name}<br>Gênero: {genre}<br>Tempo Médio de Jogo: {playtime}<br>Avaliações Negativas: {neg_ratings}'
                for name, genre, playtime, neg_ratings in
                zip(data_to_plot['app_name_df2'], data_to_plot['genres'], data_to_plot['average_playtime'], data_to_plot['negative_ratings'])],
            hovertemplate='%{text}<extra></extra>'
        ))

    fig.update_layout(
        title='',
        xaxis_title='Tempo médio de jogo',
        yaxis_title='Avaliações',
        # width=850,
        # height=500
    )
    
    st.plotly_chart(fig)
    st.write(f'''<p style='text-align: center'>
             Representação em scatter mostrando a relação entre tempo de jogo médio e quantidade de avaliações</p>
             ''', unsafe_allow_html=True)

def grafico_5(selected_games, selected_reviews, df):

    if not selected_games:  #carrega todos os jogos para comparação caso nenhum esteja selecionado
        selected_games = df['app_name'].unique()

    var_compara_selecao = compara_selecao(True, selected_games, selected_reviews)

    if var_compara_selecao[0] != 0:
        st.write(f'''<p style='text-align: center'>
                <br>{var_compara_selecao[1]}</p>
                ''', unsafe_allow_html=True)
        st.stop()

    st.write(f'''<h3 style='text-align: center'><br>
            Relação entre a polaridade média das reviews e a quantidade média de jogadores<br><br></h3>
            ''', unsafe_allow_html=True)
    
    filtered_data_2 = df[(df["app_name"].isin(selected_games))]

    #calcular a media de polaridade de reviews por jogo
    positivas = filtered_data_2.groupby('app_id')['review_score'].sum()
    reviews_totais = filtered_data_2.groupby('app_id')['review_score'].count()
    med_polaridade = ((positivas / reviews_totais) * 100).clip(lower=0)

    #filtrar os jogos baseado na média de reviews
    if 'Negativa' in selected_reviews and 'Positiva' not in selected_reviews:
        med_polaridade = med_polaridade[med_polaridade < 50]
    elif 'Positiva' in selected_reviews and 'Negativa' not in selected_reviews:
        med_polaridade = med_polaridade[med_polaridade >= 50]
    
    player_counts = []
    app_ids = []
    app_names = []
    
    #carregar os dados de contagem e jogadores
    for app_id in med_polaridade.index:
        try:
            player_data = carrega_df(app_id)
            player_count = player_data['Playercount'].mean()
            player_counts.append(player_count)
            app_name = filtered_data_2[filtered_data_2['app_id'] == app_id]['app_name'].values[0]
            app_ids.append(app_id)
            app_names.append(app_name)
        except FileNotFoundError:
            pass

    #df com a player count de cada jogo
    player_df = pd.DataFrame({'app_id': app_ids, 'app_name': app_names, 'player_count': player_counts})

    #add um slider na sidebar para remoção de outliers
    outlier_threshold = st.sidebar.slider('Descartar Outliers (Quantidade média de jogadores)', min_value=10000, max_value=999999, value=100000, step=5000)
    

    #df com a player count de cada jogo
    player_df = pd.DataFrame({'app_id': app_ids, 'app_name': app_names, 'player_count': player_counts})
    #remove outliers com mais do que o threshold
    player_df_threshold = player_df[player_df["player_count"] > outlier_threshold]
    #remove outliers com menos do que o threshold
    player_df_threshold = player_df[player_df["player_count"] <= outlier_threshold]
    
    #df com playercount e sentimentos
    merged_player_sentimentos_df = pd.merge(med_polaridade.reset_index(), player_df_threshold, on='app_id')
    
    # Apresentando erro no tratamento das colunas
    # merged_player_sentimentos_df.rename(columns = {'app_id': 'id', 'review_score': 'Avaliação do jogo', 'app_name': 'Nome do jogo',
    #                                             'player_count': 'Contagem de jogadores'}, inplace=True)

    # Exibe a tabela com o dataframe
    col1, col2, col3 = st.columns([1,5,1])

    with col1:
        pass
    with col2:
        st.dataframe(merged_player_sentimentos_df, hide_index=True,)
    with col3:
        pass
    
    fig = px.scatter(merged_player_sentimentos_df, x="review_score", y="player_count",
                        title='',
                        labels={'review_score':'Média das avaliações (%)', 'player_count':'Quantidade média de jogadores'},
                        hover_data=['app_name'],
                        color='review_score',            
                        color_continuous_scale=[(0, "red"),(1, "green")])
    st.plotly_chart(fig)
    st.write(f'''<p style='text-align: center'>
             Essa visualização mostra a relação entre a polaridade média das reviews e a contagem média de jogadores para o(s) jogo(s) selecionados.
             Tem como objetivo notar tendências dentro do dataset previamente.</p>
             ''', unsafe_allow_html=True)

def grafico_6(selected_games, selected_reviews, df):

    if not selected_games:  #carrega todos os jogos para comparação caso nenhum esteja selecionado
        selected_games = df['app_name'].unique()

    var_compara_selecao = compara_selecao(True, selected_games, selected_reviews)

    if var_compara_selecao[0] != 0:
        st.write(f'''<p style='text-align: center'>
                <br>{var_compara_selecao[1]}</p>
                ''', unsafe_allow_html=True)
        st.stop()

    st.write(f'''<h3 style='text-align: center'><br>
            Correlação entre a quantidade média de jogadores e quantidade média de avaliações indicadas como úteis<br><br></h3>
            ''', unsafe_allow_html=True)
    
    filtered_data_2 = df[(df["app_name"].isin(selected_games))]
    
    # Calcula a quantidade média de reviews indicadas como úteis por jogo
    
    reviews_indicadas = filtered_data_2.groupby('app_id')['review_votes'].sum()
    reviews_totais = filtered_data_2.groupby('app_id')['review_votes'].count()
    media_uteis = ((reviews_indicadas / reviews_totais) * 100).clip(lower=0)
        
    contagens_jogadores = []
    jogos_ids = []
    jogos_nomes = []
        
    # Carregar a quantidade de jogadores
    
    for app_id in media_uteis.index:
        try:
            dados_jogadores = carrega_df(app_id)
            contagem_jogadores = dados_jogadores['Playercount'].mean()
            contagens_jogadores.append(contagem_jogadores)
            jogo_nome = filtered_data_2[filtered_data_2['app_id'] == app_id]['app_name'].values[0]
            jogos_nomes.append(jogo_nome)
            jogos_ids.append(app_id)
        except FileNotFoundError:
            pass

    # Dataframe com a quantidade de jogadores de cada jogo
        
    jogadores_df = pd.DataFrame({'app_id': jogos_ids, 'app_name': jogos_nomes, 'player_count': contagens_jogadores})
    
    # Dataframe com a quantidade de jogadores e indicações de reviews
    
    mesclado_jogadores_indicacoes_df = pd.merge(media_uteis.reset_index(), jogadores_df, on='app_id')

    # Exibe a tabela com o dataframe
    col1, col2, col3 = st.columns([1,5,1])

    with col1:
        pass
    with col2:
        st.dataframe(mesclado_jogadores_indicacoes_df, hide_index=True,)
    with col3:
        pass

    grafvotes = px.scatter(mesclado_jogadores_indicacoes_df, x="review_votes", y="player_count",
                            title='',
                            labels={'review_votes':'Média de avaliações indicadas como úteis (%)', 'player_count':'Quantidade média de jogadores'},
                            hover_data=['app_name'],
                            color='review_votes',            
                            color_continuous_scale=[(0, "red"),(1, "green")])

    st.plotly_chart(grafvotes)
    st.write(f'''<p style='text-align: center'>
             Essa visualização mostra a relação entre a contagem média de jogadores e o número de avaliações consideradas como úteis pela comunidade para os jogos selecionados.
             Tem como objetivo mostrar tendências dentro do dataset previamente.
             </p>
             ''', unsafe_allow_html=True)

def main():
    build_header()
    build_body()

if __name__ == "__main__":
    main()
