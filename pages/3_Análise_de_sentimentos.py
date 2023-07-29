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

df = load_csv()
df_tags = load_csv2()
df_merged = pd.merge(df, df_tags, left_on=["app_id", "app_name"], right_on=["appid", "name"])


st.title("Exploração de dados apartir da análise de sentimentos")
st.markdown("---")
st.sidebar.subheader("Use os filtros para exploração de dados:")

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  -  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

df_merged["sentiment"] = df_merged["review_score"].apply(lambda x: 1 if x == 1 else 0)


#Seta os inputs e outputs para os modelos de teste e de treino
X_train, X_test, y_train, y_test = train_test_split(df_merged["review_text"], df_merged["sentiment"], test_size=0.2, random_state=42)

#Vetorizando os dados de texto
vectorizer = CountVectorizer(stop_words="english")
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

model_choice = st.sidebar.selectbox("Select a model", ["Naive Bayes", "Random Forest"])
if model_choice == "Naive Bayes":
    clf = MultinomialNB()
elif model_choice == "Random Forest":
    clf = RandomForestClassifier(n_estimators=35, max_depth=8, random_state=42, criterion="gini")
    
#Treinando o modelo
clf.fit(X_train_vect, y_train)

#Fazendo a avaliacao do modelo
accuracy = clf.score(X_test_vect, y_test)

#Dando dados do naive bayes para uma coluna no dataframe
df_merged['predicted_sentiment'] = clf.predict(vectorizer.transform(df_merged['review_text']))
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  -DASHBOARD-  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

st.sidebar.header("Filtrar por jogo")
selected_game = st.sidebar.selectbox("Selecione um jogo", sorted(df_merged["app_name"].unique()))
#Filtrar o dataset baseado no jogo selecionado
df_filtered = df_merged[df_merged["app_name"] == selected_game]

#Mostra algumas informações sobre o jogo
st.header(selected_game)
st.write("Numero total de reviews: ", len(df_filtered))
st.write("Pontuação média das reviews: ", round(df_filtered["review_score"].mean(), 2))
st.write("Percentual de reviews positivas: ", round(df_filtered["sentiment"].mean() * 100, 2), "%")

st.subheader("Histograma das labels de sentimento:")
fig = px.histogram(df_filtered, x="predicted_sentiment", nbins=2)
st.plotly_chart(fig)

#Matriz de confusão do naive bayes
st.subheader("Matriz de confusão:")
y_pred = clf.predict(X_test_vect)
cm = pd.crosstab(y_test, y_pred, rownames=["Real"], colnames=["Previsto"])
fig3, ax3 = plt.subplots(figsize=(5 ,5))
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
sns.color_palette("flare", as_cmap=True)
sns.heatmap(cm, annot=True, fmt="d")
st.pyplot(fig3)

st.write("Precisão do modelo:", accuracy)

#Pega os coeficientes das features dentro do modelo
if model_choice == "Random Forest":
    coefs = clf.feature_importances_
else:
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
    st.subheader("Word cloud de reviews positivas")
    st.image(positive_wordcloud.to_image())
else:
    st.subheader("Word cloud de reviews positivas")
    st.write("Nenhuma review positiva encontrada para este jogo.")

if len(negative_text) > 0:
    negative_wordcloud = WordCloud(width=800, height=400, background_color="black", max_words=25).generate_from_text(negative_text)
    st.subheader("Word cloud de reviews negativas")
    st.image(negative_wordcloud.to_image())
else:
    st.subheader("Word cloud de reviews negativas")
    st.write("Nenhuma review negativa encontrada para este jogo.")

try:
    #Carrega os dados de contagem de jogadores para o jogo selecionado
    gameid = df_filtered["appid"].iloc[0]
    df_playercount = load_csv3(gameid)
    
    #Agrega os dados de contagem de jogadores por data (média por horas)
    df_playercount['Time'] = pd.to_datetime(df_playercount['Time'])
    df_playercount.set_index('Time', inplace=True)
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
   st.write('A pontuação geral de sentimento prevista para ',selected_game,'é de:', round(overall_sentiment*100,2),'% vs a verdadeira pontuação das reviews de:',  round(df_filtered["sentiment"].mean() * 100, 2), "%")

