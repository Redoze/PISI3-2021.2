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


#seta os inputs e outputs para os modelos de teste e de treino
X_train, X_test, y_train, y_test = train_test_split(df_merged["review_text"], df_merged["sentiment"], test_size=0.2, random_state=42)

#vetorizando os dados de texto
vectorizer = CountVectorizer(stop_words="english")
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

model_choice = st.sidebar.selectbox("Select a model", ["Naive Bayes", "Random Forest"])
if model_choice == "Naive Bayes":
    clf = MultinomialNB()
elif model_choice == "Random Forest":
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
#treinando o Naive Bayes
clf.fit(X_train_vect, y_train)

#fazendo a avaliacao do modelo
accuracy = clf.score(X_test_vect, y_test)

#dando dados do naive bayes para uma coluna no dataframe
df_merged['predicted_sentiment'] = clf.predict(vectorizer.transform(df_merged['review_text']))
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  -DASHBOARD-  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

st.sidebar.header("Filtrar por jogo")
selected_game = st.sidebar.selectbox("Selecione um jogo", sorted(df_merged["app_name"].unique()))
#filtrar o dataset baseado no jogo selecionado
df_filtered = df_merged[df_merged["app_name"] == selected_game]

#mostra algumas informações sobre o jogo
st.header(selected_game)
st.write("Numero total de reviews: ", len(df_filtered))
st.write("Pontuação média das reviews: ", round(df_filtered["review_score"].mean(), 2))
st.write("Percentual de reviews positivas: ", round(df_filtered["sentiment"].mean() * 100, 2), "%")

st.subheader("Histograma das labels de sentimento")
fig = px.histogram(df_filtered, x="predicted_sentiment", nbins=2)
st.plotly_chart(fig)

#matriz de confusão do naive bayes
y_pred = clf.predict(X_test_vect)
cm = pd.crosstab(y_test, y_pred, rownames=["Verdadeira"], colnames=["Previsão"])
fig3, ax3 = plt.subplots()
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
sns.color_palette("flare", as_cmap=True)
sns.heatmap(cm, annot=True, fmt="d")
st.pyplot(fig3)

st.write("Precisão do modelo:", accuracy)

#pega os coeficientes das features dentro do modelo
coefs = clf.feature_importances_ if model_choice == "Random Forest" else clf.feature_log_prob_
#pega o nome das features pelo vetorizador
feature_names = vectorizer.get_feature_names_out()
#ordena os nomes das features por seus coeficientes para sentimentos positivos
positive_features = [feature_names[i] for i in np.argsort(coefs[0])[::-1][:100]]
#ordena os nomes das features por seus coeficientes para sentimentos negativos
negative_features = [feature_names[i] for i in np.argsort(coefs[0])[:100]]
#criar o wordcloud pra reviews positivas e negativas
positive_text = " ".join(df_filtered[df_filtered["predicted_sentiment"] == 1]["review_text"])
negative_text = " ".join(df_filtered[df_filtered["predicted_sentiment"] == 0]["review_text"])

positive_wordcloud = WordCloud(width=800, height=400, background_color="black", max_words=25).generate_from_text(positive_text)
negative_wordcloud = WordCloud(width=800, height=400, background_color="black", max_words=25).generate_from_text(negative_text)

st.subheader("Word cloud de reviews positivas")
st.image(positive_wordcloud.to_image())

st.subheader("Word cloud de reviews negativas")
st.image(negative_wordcloud.to_image())

#correlation heatmap
#cols_to_include = ["average_playtime", "review_score", "review_votes"]
#df_selected = df_filtered[cols_to_include]
#corr_matrix = df_selected.corr()

# Set up the mask to hide the upper triangle
#mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the color map
#cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Generate a heatmap of the correlation matrix
#fig, ax = plt.subplots(figsize=(10,10))
#sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .5})
#ax.set_title('Correlation Heatmap')
#st.pyplot(fig)
