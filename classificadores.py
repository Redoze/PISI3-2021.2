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
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import compute_class_weight
from sklearn.neighbors import KNeighborsClassifier

# Definindo a configuração da página
st.set_page_config(
    page_title="Análise de sentimentos",
    page_icon="🔎",
    layout="centered",
)

# Função para construir o cabeçalho
def build_header():
    # Cabeçalho com título e descrição
    st.write(f'''<h1 style='text-align: center'
             >Exploração de dados apartir da análise de sentimentos<br></h1>
             ''', unsafe_allow_html=True)
    
    st.write(f'''<p style='text-align: center'>
            <br>Nessa página nós fazemos os testes de vários algoritmos de classificação que usaremos mais tarde no simulador de jogos para identificar
            as keywords mais importantes nas avaliações.<br></p>
            ''', unsafe_allow_html=True)
    st.markdown("---")

# Função para construir o corpo principal
def build_body():
    st.write(f'''<h2 style='text-align: center; font-size: 36px'>
            Resultados da Análise de sentimentos<br><br></h2>
             ''', unsafe_allow_html=True) # 36px equivalem ao h2/subheader
    
    df = carrega_df('df1')
    game_options = df["app_name"].dropna().unique() # Adicionado o método 'dropna()' para remover os valores nulos.
    model_options = {"Naive Bayes": 'naive', "k-Nearest Neighbor": 'k_nearest', "Support Vector Machine": 'support_vector', "Regressão Logística": 'regressao_logistica'}

    df["sentiment"] = df["review_score"].apply(lambda x: 1 if x == 1 else 0)

    # Função para iniciar o modelo selecionado
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
        selected_game = selectbox("Selecione um jogo", game_options)
        selected_game = [selected_game] 
        # Após a mudança do uso do selectbox do streamlit extras que possui (---), foi preciso transformar selected_game para 
        # lista a fim de ser compatível com o método 'isin()' da função 'inicia_modelo'.

    with coluna_2:
        selected_model = selectbox("Selecione o modelo de classificação", list(model_options.keys()))

    with vazio_2:
        st.empty()
    
    if selected_game[0] != None and selected_model != None:

        st.write(f'''<h3 style='text-align: center'>
                <br>Classificação {selected_model} para o jogo {selected_game[0]}<br><br></h3>
                ''', unsafe_allow_html=True)
        
        inicia_modelo(0)
    
    else:
        st.write(f'''<p style='text-align: center'>
                 <br>Utilize as caixas de seleção acima para selecionar os dados para análise de sentimento.</p>
                ''', unsafe_allow_html=True)
    
############################################################ Modelos ############################################################
    
def naive_bayes(selected_game, df_filtered):
    
    
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
    
    y_pred = clf.predict(X_test_vect)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Exibindo métricas de avaliação
    scores = cross_val_score(clf,X=X_train_vect,y=y_train,cv=5)
    mean_accuracy = np.mean(scores)
    std_deviation = scores.std()

    
    try:
        #Carrega os dados de contagem de jogadores para o jogo selecionado
        gameid = df_filtered["app_id"].iloc[0]
        df_playercount = carrega_df(gameid)
        
        #Agrega os dados de contagem de jogadores por data (média por horas)
        df_playercount.reset_index()
        
        #df_playercount['Time'] = pd.to_datetime(df_playercount['Time'])
        #df_playercount.set_index('Time', inplace=True)
        daily_pc_df_tmp= df_playercount.resample('D').mean().reset_index()
        
    except (FileNotFoundError, TypeError):
        pass
    else: 
        #Plot da contagem média de jogadores diarios
        st.subheader('Média de jogadores diários:')
        fig_daily_pc = px.area(daily_pc_df_tmp, x='Time', y='Playercount')
        st.plotly_chart(fig_daily_pc)
            
        #Computa a pontuacao de sentimento geral para o jogo selecionado
        overall_sentiment = df_filtered['predicted_sentiment'].mean()
       
        st.subheader("Sentimento geral previsto:")
        st.write('A pontuação geral de sentimento prevista para ',f'{selected_game[0]}','é de:', round(overall_sentiment*100,2),'% vs a verdadeira pontuação das reviews de:',  round(df_filtered["sentiment"].mean() * 100, 2), "%")

    return[accuracy, recall, precision, f1, mean_accuracy, std_deviation]

def k_nearest(selected_game, df_filtered):

    # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(df_filtered["review_text"], df_filtered["sentiment"], test_size=0.2, random_state=42)

    # Convertendo as palavras em recursos numéricos (vetores TF-IDF)
    vectorizer = TfidfVectorizer(max_features = 100) 
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    def predict(x, X_train, y_train, k=3):
        distances = [np.linalg.norm(x - x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train.iloc[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    # Treinando o modelo e fazendo previsões
    predictions = [predict(x, X_train_tfidf.toarray(), y_train, k=3) for x in X_test_tfidf.toarray()]

    # Adiciona as previsões mapeadas ao DataFrame
    df_predicted = df_filtered.loc[y_test.index].copy()
    df_predicted['predicted_sentiment'] = predictions

    y_pred = predictions

    # Avalia o desempenho do modelo
    accuracy = accuracy_score(y_test, predictions)

    # Calcula as métricas
    recall = recall_score(y_test, predictions, average='weighted')
    precision = precision_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    # Calcula a acurácia média usando cross-validation
    cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=3), X_train_tfidf, y_train, cv=5)
    mean_cv_accuracy = np.mean(cv_scores)
    std_cv_accuracy = np.std(cv_scores)
    
    return[accuracy, recall, precision, f1, mean_cv_accuracy, std_cv_accuracy]

def support_vector_machine(selected_game, df_filtered):

    # Concatenar as colunas de review_text e review_score
    df_filtered["combined_features"] = df_filtered["review_text"] + " " + df_filtered["review_score"].astype(str)

    # Balancear os dados para incluir tanto reviews positivas quanto negativas
    num_samples_per_class = 200
    num_negative_samples = min(len(df_filtered[df_filtered["sentiment"] == 0]), num_samples_per_class)

    # Ajuste do número de amostras por classe com base no tamanho da classe minoritária
    if num_negative_samples < num_samples_per_class:
        num_samples_per_class = num_negative_samples

    positive_samples = df_filtered[df_filtered["sentiment"] == 1].sample(n=num_samples_per_class, random_state=42)
    negative_samples = df_filtered[df_filtered["sentiment"] == 0].sample(n=num_samples_per_class, random_state=42)
    balanced_df = pd.concat([positive_samples, negative_samples])

    # Dividir os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(balanced_df["combined_features"], balanced_df["sentiment"],
                                                        test_size=0.2, random_state=42)

    # Vetorização dos textos usando TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Criação e treinamento do modelo SVM
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train_tfidf, y_train)

    # Fazendo previsões usando o modelo treinado
    predictions = clf.predict(X_test_tfidf)

    # Criando um DataFrame para armazenar as previsões
    df_predicted = balanced_df.loc[y_test.index].copy()
    df_predicted['predicted_sentiment'] = predictions

    # Cálculo de métricas de avaliação
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Acurácia média do modelo usando cross-validation
    scores = cross_val_score(clf, X=X_train_tfidf, y=y_train, cv=5)
    mean_accuracy = np.mean(scores)
    std_deviation = scores.std()

    return[accuracy, recall, precision, f1, mean_accuracy, std_deviation]
	
def regressao_logistica(selected_game, df_filtered):

    # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(df_filtered["review_text"], df_filtered["sentiment"],
                                                        test_size=0.2, random_state=42)

    # Vetorização dos textos usando TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Calculando os pesos das classes
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)

    # Criação e treinamento do modelo de Regressão Logística com pesos ajustados
    clf = LogisticRegression(random_state=42, class_weight={0: class_weights[0], 1: class_weights[1]})
    clf.fit(X_train_tfidf, y_train)

    # Fazendo previsões usando o modelo treinado
    predictions = clf.predict(X_test_tfidf)

    # Criando um DataFrame para armazenar as previsões
    df_predicted = df_filtered.loc[y_test.index].copy()
    df_predicted['predicted_sentiment'] = predictions

    # Cálculo de métricas de avaliação
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Acurácia média do modelo usando cross-validation
    scores = cross_val_score(clf, X=X_train_tfidf, y=y_train, cv=5)
    mean_accuracy = np.mean(scores)
    std_deviation = scores.std()

    return[accuracy, recall, precision, f1, mean_accuracy, std_deviation]

def main():
    build_header()
    build_body()

if __name__ == "__main__":
    main()
