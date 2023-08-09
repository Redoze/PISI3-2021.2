import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
from funcs import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

st.set_page_config(
    page_title="Análise de sentimentos",
    page_icon="🔎",
    layout="centered",
)

def build_header():

    st.write(f'''<h1 style='text-align: center'
             >Exploração de dados apartir da análise de sentimentos<br><br></h1>
             ''', unsafe_allow_html=True)
    
    st.write(f'''<h2 style='text-align: center; font-size: 18px'>
    Análise de sentimentos das reviews sob as colunas do dataset.<br></h2>
        ''', unsafe_allow_html=True)
    st.markdown("---")


def build_body():

    st.write(f'''<h2 style='text-align: center; font-size: 36px'>
            Resultados da Análise de sentimentos</h2>
             ''', unsafe_allow_html=True) # 36px equivalem ao h2/subheader
    st.text("")
    
    df = carrega_df('df1')
    df_tags = carrega_df('df2')
    game_options = df["app_name"].unique()
    model_options = {"Naive Bayes": 'modelo_1'}

    
    df_merged = pd.merge(df, df_tags, left_on=["app_id", "app_name"], right_on=["app_id_df2", "app_name_df2"])
    df_merged["sentiment"] = df_merged["review_score"].apply(lambda x: 1 if x == 1 else 0)

    def inicia_modelo(posicao):
        #Filtrar o dataset baseado no jogo selecionado
        df_filtered = df_merged[(df_merged["app_name"].isin(selected_game))]
        
        for nome_funcao, modelos in model_options.items():
            
            if nome_funcao == selected_model:
                #st.write(globals())
                chama_funcao = globals()[modelos]
                chama_funcao(df, selected_game, selected_model, df_filtered)

    ############################################################ - ############################################################                

    vazio_1, coluna_1, coluna_2, vazio_2 = st.columns([1,3,3,1])
    
    with vazio_1:
        st.empty()

    with coluna_1:
        selected_game = [st.selectbox("Selecione um jogo", game_options)]

    with coluna_2:
        selected_model = st.selectbox("Selecione o modelo de classificação", list(model_options.keys()))

    with vazio_2:
        st.empty()

    ############################################################ Exibição dos modelos ############################################################

    vazio_1_lv_3, coluna_1_lv_3, vazio_2_lv_3 = st.columns([1,18,1])
    
    with vazio_1_lv_3:
        st.empty()

    with coluna_1_lv_3:
        inicia_modelo(0)
            
    with vazio_2_lv_3:
        st.empty()         
        
############################################################ Modelos ############################################################
    
def modelo_1(df, selected_game, selected_model, df_filtered):

    st.write(f'''<h3 style='text-align: center'><br>
    Naive Bayes<br><br></h3>
        ''', unsafe_allow_html=True)
    
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

    
    #Mostra algumas informações sobre o jogo
    st.write(f'''<h2 style='text-align: center; font-size: 36px'>
            {selected_game[0]} </h2>
             ''', unsafe_allow_html=True)
    st.text("")
    
    st.write("Numero total de reviews: ", len(df_filtered))
    st.write("Pontuação média das reviews: ", round(df_filtered["review_score"].mean(), 2))
    st.write("Percentual de reviews positivas: ", round(df_filtered["sentiment"].mean() * 100, 2), "%")
    st.text("")
    
    st.subheader("Histograma das labels de sentimento:")
    fig = px.histogram(df_filtered, x="predicted_sentiment", nbins=2, labels=dict(predicted_sentiment="Polaridade Prevista", count="Contagem de Reviews"))
    fig.update_xaxes(
        ticktext=["Negativas", "Positivas"],
        tickvals=[0,1]
    )
    st.plotly_chart(fig)
    
    #Matriz de confusão do naive bayes
    st.subheader("Matriz de confusão:")
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
    
    st.write(f'''<h2 style='text-align: center; font-size: 26px'>
            Acurácia do modelo: {accuracy} </h2>
             ''', unsafe_allow_html=True)
    st.text("")
    st.text("")
    
    #Pega os coeficientes das features dentro do modelo
    coefs = clf.feature_log_prob_
    #Pega o nome das features pelo vetorizador
    feature_names = vectorizer.get_feature_names_out()
    #Pega os labels de classe
    classes = clf.classes_
    #Pega o indice da classe negativa
    neg_index = np.where(classes == 0)[0][0]
    #Pega o indice da classe positiva
    pos_index = np.where(classes == 1)[0][0]
    #Extrai os atributos negativos
    negative_features = [feature_names[i] for i in np.argsort(coefs[neg_index])[:100]]
    #Extrai os atributos positivos
    positive_features = [feature_names[i] for i in np.argsort(coefs[pos_index])[::-1][:100]]
    
    
    #Criar o wordcloud pra reviews positivas e negativas
    positive_text = " ".join(df_filtered[df_filtered["predicted_sentiment"] == 1]["review_text"])
    negative_text = " ".join(df_filtered[df_filtered["predicted_sentiment"] == 0]["review_text"])
    
    # Verificar se o texto não está vazio antes de criar a nuvem de palavras
    if len(positive_text) > 0:
        positive_wordcloud = WordCloud(width=800, height=400, background_color="black", max_words=25).generate_from_text(positive_text)
        st.subheader("Word cloud de reviews positivas:")
        st.image(positive_wordcloud.to_image())
    else:
        st.subheader("Word cloud de reviews positivas:")
        st.write("Nenhuma review positiva encontrada para este jogo.")
    
    if len(negative_text) > 0:
        negative_wordcloud = WordCloud(width=800, height=400, background_color="black", max_words=25).generate_from_text(negative_text)
        st.subheader("Word cloud de reviews negativas:")
        st.image(negative_wordcloud.to_image())
    else:
        st.subheader("Word cloud de reviews negativas:")
        st.write("Nenhuma review negativa encontrada para este jogo.")
    
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

def main():
    build_header()
    build_body()

if __name__ == "__main__":
    main()
