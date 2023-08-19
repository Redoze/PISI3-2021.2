import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from funcs import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from wordcloud import WordCloud
from streamlit_extras.no_default_selectbox import selectbox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from collections import Counter

st.set_page_config(
    page_title="Análise de sentimentos",
    page_icon="🔎",
    layout="centered",
)

def build_header():

    st.write(f'''<h1 style='text-align: center'
             >Exploração de dados apartir da análise de sentimentos<br></h1>
             ''', unsafe_allow_html=True)
    
    st.write(f'''<h2 style='text-align: center; font-size: 18px'>
            <br>Análise de sentimentos das avaliações sob as colunas do conjuto de dados.<br></h2>
        ''', unsafe_allow_html=True)
    st.markdown("---")

def build_body():

    st.write(f'''<h2 style='text-align: center; font-size: 36px'>
            Resultados da Análise de sentimentos<br><br></h2>
             ''', unsafe_allow_html=True) # 36px equivalem ao h2/subheader
    
    df = carrega_df('df1')
    # Removido carregamento e processamento do df2 que estava aqui. Era realmente necessário?

    game_options = df["app_name"].unique()
    model_options = {"Naive Bayes": 'modelo_1', "K-Nearest Neighbor": 'modelo_2'}

    df["sentiment"] = df["review_score"].apply(lambda x: 1 if x == 1 else 0)

    def inicia_modelo(posicao):
        #Filtrar o dataset baseado no jogo selecionado
        df_filtered = df[(df["app_name"].isin(selected_game))]
        del df_filtered['app_id']
        del df_filtered['app_name']
        del df_filtered['review_votes']

        for nome_funcao, modelos in model_options.items():
            
            if nome_funcao == selected_model:
                #st.write(globals())
                chama_funcao = globals()[modelos]
                # Removido os argumentos "df" e "selected_model" que não eram utilizados, caso os próximos modelo utilizem eles,
                # é melhor ser implementada condicionais especificas da chama_funcao para cada modelo.
                chama_funcao(selected_game, df_filtered)

    ############################################################ - ############################################################                

    vazio_1, coluna_1, coluna_2, vazio_2 = st.columns([1,3,3,1])
    
    with vazio_1:
        st.empty()

    with coluna_1:
        selected_game = [st.selectbox("Selecione um jogo", game_options)]

    with coluna_2:
        selected_model = selectbox("Selecione o modelo de classificação", list(model_options.keys()))

    with vazio_2:
        st.empty()
    
    inicia_modelo(0)

############################################################ Modelos ############################################################
    
def modelo_1(selected_game, df_filtered):

    st.write(f'''<h3 style='text-align: center'>
             <br>Classificação Naive Bayes para o jogo {selected_game[0]}<br><br></h3>
        ''', unsafe_allow_html=True)
    
    st.write(f'''<p style='text-align: center'>
             PLACEHOLDERPLACEHOLDERPLACEHOLDERPLACEHOLDER<br>PLACEHOLDER SOBRE O CLASSIFICADOR <br>PLACEHOLDERPLACEHOLDERPLACEHOLDERPLACEHOLDER<br><br></p>
             ''', unsafe_allow_html=True)
    
    # Função que mostra em tela as estatiscas das avaliações do jogo
    informacoes_sobre_jogo(df_filtered)
    
    #Seta os inputs e outputs para os modelos de teste e de treino
    X_train, X_test, y_train, y_test = train_test_split(df_filtered["review_text"], df_filtered["sentiment"], test_size=0.2, random_state=42)
    
    #Vetorizando os dados de texto
    vectorizer = CountVectorizer(stop_words="english")
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    
    clf = MultinomialNB()
        
    #Treinando o modelo
    clf.fit(X_train_vect, y_train)
    
    #Fazendo a avaliacao do modelo
    accuracy = clf.score(X_test_vect, y_test)
    
    #Dando dados do naive bayes para uma coluna no dataframe
    df_filtered['predicted_sentiment'] = clf.predict(vectorizer.transform(df_filtered['review_text']))

    st.write(f'''<h3 style='text-align: center'>
            Histograma das avalições por sentimento</h3>
            ''', unsafe_allow_html=True)
    
    fig = px.histogram(df_filtered, x="predicted_sentiment", nbins=2, labels=dict(predicted_sentiment="Polaridade prevista ", count="Contagem de avaliações "))
    fig.update_xaxes(
        ticktext=[" Negativas", " Positivas"],
        tickvals=[0,1]
    )
    st.plotly_chart(fig)
    
    #Matriz de confusão do naive bayes
    st.write(f'''<h3 style='text-align: center'>
        Matriz de confusão<br><br></h3>
        ''', unsafe_allow_html=True)
    
    y_pred = clf.predict(X_test_vect)
    cm = pd.crosstab(y_test, y_pred, rownames=["Real"], colnames=["Previsto"])
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")
    sns.color_palette("flare", as_cmap=True)
    sns.heatmap(cm, annot=True, fmt="d")
    ax3.set_xticklabels(["Negativas", "Positivas"], fontsize=11)
    ax3.set_yticklabels(["Negativas", "Positivas"], fontsize=11)
    st.pyplot(fig3)
    
    st.write(f'''<p style='text-align: center'>
            <br>Acurácia do modelo: {accuracy:.2f} </p>
             ''', unsafe_allow_html=True)
    st.text("")
    scores = cross_val_score(clf,X=X_train_vect,y=y_train,cv=5)

    mean_accuracy = np.mean(scores)
    std_deviation = scores.std()

    st.write(f'''<p style='text-align: center'>
            Acurácia média do modelo usando cross-validation: {mean_accuracy:.2f}
            Desvio padrão de: {std_deviation:.2f}</p>
             ''', unsafe_allow_html=True)
    st.text("")
    
    # #Pega os coeficientes das features dentro do modelo
    # coefs = clf.feature_log_prob_
    # #Pega o nome das features pelo vetorizador
    # feature_names = vectorizer.get_feature_names_out()
    # #Pega os labels de classe
    # classes = clf.classes_
    # #Pega o indice da classe negativa
    # neg_index = np.where(classes == 0)[0][0]
    # #Pega o indice da classe positiva
    # pos_index = np.where(classes == 1)[0][0]
    # #Extrai os atributos negativos
    # negative_features = [feature_names[i] for i in np.argsort(coefs[neg_index])[:100]]
    # #Extrai os atributos positivos
    # positive_features = [feature_names[i] for i in np.argsort(coefs[pos_index])[::-1][:100]]
    
    #Criar o wordcloud pra reviews positivas e negativas
    positive_text = " ".join(df_filtered[df_filtered["predicted_sentiment"] == 1]["review_text"])
    negative_text = " ".join(df_filtered[df_filtered["predicted_sentiment"] == 0]["review_text"])
    
    # Verificar se o texto não está vazio antes de criar a nuvem de palavras
    if len(positive_text) > 0:
        positive_wordcloud = WordCloud(width=800, height=400, background_color="black", max_words=25).generate_from_text(positive_text)
        st.write(f'''<h3 style='text-align: center'>
                 Nuvem de palavras de avaliações positivas:<br><br></h3>
                 ''', unsafe_allow_html=True)
        st.image(positive_wordcloud.to_image())
        st.write("")
    else:
        st.write(f'''<h3 style='text-align: center'>
                Nuvem de palavras de avaliações positivas:<br><br></h3>
                ''', unsafe_allow_html=True)
        st.write(f'''<p style='text-align: center'>
                Nenhuma review positiva encontrada para este jogo.<br></p>
                ''', unsafe_allow_html=True)
        st.write("")
    
    if len(negative_text) > 0:
        negative_wordcloud = WordCloud(width=800, height=400, background_color="black", max_words=25).generate_from_text(negative_text)
        st.write(f'''<h3 style='text-align: center'>
                 Nuvem de palavras de avaliações negativas:<br><br></h3>
                 ''', unsafe_allow_html=True)
        st.image(negative_wordcloud.to_image())
        st.write("")
    else:
        st.write(f'''<h3 style='text-align: center'>
                Nuvem de palavras de avaliações negativas:<br><br></h3>
                ''', unsafe_allow_html=True)
        st.write(f'''<p style='text-align: center'>
                Nenhuma review negativa encontrada para este jogo.<br></p>
                ''', unsafe_allow_html=True)
        st.write("")
    
    # Código parado temporariamente para evitar a exibição de erros
    st.stop()
    
    try:
        #Carrega os dados de contagem de jogadores para o jogo selecionado
        gameid = df_filtered["app_id"].iloc[0]
        df_playercount = carrega_df(gameid)
        
        #Agrega os dados de contagem de jogadores por data (média por horas)
        df_playercount.reset_index()
        
        #df_playercount['Time'] = pd.to_datetime(df_playercount['Time'])
        #df_playercount.set_index('Time', inplace=True)
        daily_pc_df_tmp= df_playercount.resample('D').mean().reset_index()
        
    except FileNotFoundError:
        st.write(f"Sem dados de 'jogadores diários' encontrados para o jogo: {gameid}")
    else: 
       #Plot da contagem média de jogadores diarios
       st.subheader('Média de jogadores diários:')
       fig_daily_pc = px.area(daily_pc_df_tmp, x='Time', y='Playercount')
       st.plotly_chart(fig_daily_pc)
    
       #Computa a pontuacao de sentimento geral para o jogo selecionado
       overall_sentiment = df_filtered['predicted_sentiment'].mean()
       
       st.subheader("Sentimento geral previsto:")
       st.write('A pontuação geral de sentimento prevista para ',f'{selected_game[0]}','é de:', round(overall_sentiment*100,2),'% vs a verdadeira pontuação das reviews de:',  round(df_filtered["sentiment"].mean() * 100, 2), "%")

def modelo_2(selected_game, df_filtered):

    st.write(f'''<h3 style='text-align: center'>
             <br>Classificação K-Nearest Neighbor para o jogo {selected_game[0]}<br><br></h3>
            ''', unsafe_allow_html=True)
    
    st.write(f'''<p style='text-align: center'>
             O classificador k-Nearest Neighbors (k-NN) classifica um ponto de dados com base na maioria das classes dos seus k vizinhos mais próximos em um espaço de recursos. 
             Ele não faz suposições sobre a distribuição dos dados, mas pode ser sensível à escala. Nesta análise de sentimentos, seus recursos estão definidos para 100 e o valor de k vizinhos para 3.<br><br>
             ''', unsafe_allow_html=True)
    
    # Função que mostra em tela as estatiscas das avaliações do jogo
    informacoes_sobre_jogo(df_filtered)
    
    # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(df_filtered["review_text"], df_filtered["sentiment"], test_size=0.2, random_state=42)

    # Convertendo as palavras em recursos numéricos (vetores TF-IDF)
    vectorizer = TfidfVectorizer(max_features = 100) 
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    def predict(x, X_train, y_train, k = 3):
        distances = [np.linalg.norm(x - x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train.iloc[i] for i in k_indices] # .iloc para acessar por índices
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    # Treinando o modelo e fazendo previsões
    predictions = [predict(x, X_train_tfidf.toarray(), y_train, k=3) for x in X_test_tfidf.toarray()]

    # Avaliando o desempenho do modelo
    accuracy = accuracy_score(y_test, predictions)

    st.write(f'''<p style='text-align: center'>
             <br>Acurácia do modelo: {accuracy:.2f} </p>
             ''', unsafe_allow_html=True)
    st.text("")

def informacoes_sobre_jogo(df):
    st.write(f'''<p style='text-align: center'>
            Número total de avaliações: {len(df)}<br></p>
            ''', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f'''<p style='text-align: center'>
                Pontuação média das avaliações: {round(df["review_score"].mean(), 2)} </p>
                ''', unsafe_allow_html=True)
    with col2:
        st.write(f'''<p style='text-align: center'>
                Percentual de avaliações positivas: {round(df["sentiment"].mean() * 100, 2)}%<br><br></p>
                ''', unsafe_allow_html=True)

def main():
    build_header()
    build_body()

if __name__ == "__main__":
    main()
