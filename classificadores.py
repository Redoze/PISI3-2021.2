import pandas as pd
import numpy as np
from funcs import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import compute_class_weight
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

def naive(df_filtered):

    # Define os inputs e outputs para os modelos de teste e de treino
    X_train, X_test, y_train, y_test = train_test_split(df_filtered["review_text"], df_filtered["sentiment"], test_size=0.2, random_state=42)
    
    # Vetorizando os dados de texto
    vectorizer = CountVectorizer(stop_words="english")
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    
    clf = MultinomialNB()
        
    # Treinando o modelo
    clf.fit(X_train_vect, y_train)
    
    # Fazendo a avaliação do modelo
    accuracy = clf.score(X_test_vect, y_test)
    
    # Dando dados do Naive Bayes para uma coluna no dataframe
    df_filtered['predicted_sentiment'] = clf.predict(vectorizer.transform(df_filtered['review_text']))

    # Chama a função matriz_confusao para exibir a matriz de confusão
    y_pred = clf.predict(X_test_vect)

    matriz_confusao_y_test = y_test
    matriz_confusao_y_pred = y_pred

    # Adiciona as previsões mapeadas ao DataFrame
    df_predicted = df_filtered.loc[y_test.index].copy()
    df_predicted['predicted_sentiment'] = y_pred

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Exibindo métricas
    scores = cross_val_score(clf, X=X_train_vect, y=y_train, cv=5)
    mean_accuracy = np.mean(scores)
    std_deviation = scores.std()

    return(accuracy, recall, precision, f1, mean_accuracy, std_deviation, df_predicted, matriz_confusao_y_test, matriz_confusao_y_pred)

def k_nearest(df_filtered):

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

    matriz_confusao_y_test = y_test
    matriz_confusao_y_pred = predictions

    # Adiciona as previsões mapeadas ao DataFrame
    df_predicted = df_filtered.loc[y_test.index].copy()
    df_predicted['predicted_sentiment'] = predictions

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

    return(accuracy, recall, precision, f1, mean_cv_accuracy, std_cv_accuracy, df_predicted, matriz_confusao_y_test, matriz_confusao_y_pred)

def support_vector(df_filtered):

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

    matriz_confusao_y_test = y_test
    matriz_confusao_y_pred = predictions

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

    return(accuracy, recall, precision, f1, mean_accuracy, std_deviation, df_predicted, matriz_confusao_y_test, matriz_confusao_y_pred)

def regressao_logistica(df_filtered):

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

    matriz_confusao_y_test = y_test
    matriz_confusao_y_pred = predictions

    # Cálculo de métricas de avaliação
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Acurácia média do modelo usando cross-validation
    scores = cross_val_score(clf, X=X_train_tfidf, y=y_train, cv=5)
    mean_accuracy = np.mean(scores)
    std_deviation = scores.std()
    
    return(accuracy, recall, precision, f1, mean_accuracy, std_deviation, df_predicted, matriz_confusao_y_test, matriz_confusao_y_pred)

def xgboost(df_filtered):
    
    # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(df_filtered["review_text"], df_filtered["sentiment"], test_size=0.2, random_state=42)

    # Vetorização dos textos usando TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Classificador do XGBoost
    xgb_classifier = xgb.XGBClassifier()

    # Criação e treinamento do modelo XGBoost
    xgb_classifier.fit(X_train_tfidf, y_train)

    # Fazendo previsões usando o modelo treinado
    predictions = xgb_classifier.predict(X_test_tfidf)

    # Criando um DataFrame para armazenar as previsões
    df_predicted = df_filtered.loc[y_test.index].copy()
    df_predicted['predicted_sentiment'] = predictions
    matriz_confusao_y_test = y_test
    matriz_confusao_y_pred = predictions

    # Avaliando o modelo
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    scores = cross_val_score(xgb_classifier, X=X_train_tfidf, y=y_train, cv=5)
    mean_accuracy = np.mean(scores)
    std_deviation = scores.std()
    
    
    return(accuracy, recall, precision, f1, mean_accuracy, std_deviation, df_predicted, matriz_confusao_y_test, matriz_confusao_y_pred) 
