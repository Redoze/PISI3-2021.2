import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from funcs import *
from wordcloud import WordCloud
from streamlit_extras.no_default_selectbox import selectbox
from sklearn.metrics import confusion_matrix
import classificadores

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
    model_options = {
        "Naive Bayes": ('naive',
                        "O Classificador Naive Bayes é um modelo probabilístico de aprendizado de máquina comumente utilizado para tarefas como classificação de texto, filtragem de spam e análise de sentimento. Ele aplica o Teorema de Bayes com suposições de independência ingênuas entre as características."),
        "k-Nearest Neighbor": ('k_nearest',
                               "O classificador k-Nearest Neighbors (k-NN) classifica um ponto de dados com base na maioria das classes de seus k vizinhos mais próximos em um espaço de recursos. Ele não faz suposições sobre a distribuição dos dados, embora possa ser sensível à escala. Nesta análise, os recursos estão limitados em 100 e o valor k de vizinhos próximos é definido como 3."),
        "Support Vector Machine": ('support_vector',
                                   "O Support Vector Machine (SVM) é um classificador que procura encontrar um hiperplano de separação entre diferentes classes, maximizando a margem entre os pontos de dados e o hiperplano. Nesta análise, o kernel linear é usado."),
        "Regressão Logística": ('regressao_logistica',
                                "A regressão logística é um algoritmo de classificação que utiliza a função logística para modelar a probabilidade de um evento ocorrer. Neste exemplo, estamos usando a regressão logística para análise de sentimentos."),
        "XGBoost": ('xgboost',
                    "O XGBoost (Extreme Gradient Boosting) é um poderoso algoritmo de aprendizado por ensemble que pode ser utilizado em uma ampla variedade de tarefas de classificação, incluindo análise de sentimentos."),
        "Redes Neurais": ('redes_neurais',
                          "Redes neurais são modelos de aprendizado profundo que podem aprender representações complexas de dados. Neste exemplo, usamos uma rede neural para análise de sentimentos."),
        "Random Forest": ('random_forest',
                          "O modelo Random Forest é um algoritmo de aprendizado por ensemble que utiliza múltiplas árvores de decisão para realizar a classificação. É eficaz em uma variedade de tarefas de classificação e pode ser uma escolha sólida para análise de sentimentos.")
    }

    df["sentiment"] = df["review_score"].apply(lambda x: 1 if x == 1 else 0)

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
        
        descricao = ''
        for nome_funcao, valor in model_options.items():

            if nome_funcao == selected_model:

                descricao = valor[1]

                df_filtered = df[(df["app_name"].isin(selected_game))]

                # Removido os argumentos "df" e "selected_model" que não eram utilizados, caso os próximos modelo utilizem eles,
                # é melhor ser implementada condicionais especificas da chama_funcao para cada modelo.
                chama_funcao = getattr(classificadores, valor[0]) #Anteriormente era usado "globals()"
                modelo_usado = chama_funcao(df_filtered)

                break
   
        accuracy = modelo_usado[0]
        recall = modelo_usado[1]
        precision = modelo_usado[2]
        f1 = modelo_usado[3]
        mean_accuracy = modelo_usado[4] 
        std_deviation = modelo_usado[5] 
        df_predicted = modelo_usado[6]
        matriz_confusao_y_test = modelo_usado[7] 
        matriz_confusao_y_pred = modelo_usado[8]

        st.write(f'''<p style='text-align: center'>
                {descricao}<br>''', unsafe_allow_html=True)

        informacoes_sobre_jogo(df_filtered)

        grafico_avaliacoes_sentimento(df_predicted)

        matriz_confusao(matriz_confusao_y_test, matriz_confusao_y_pred)

        informacoes_classificador(accuracy, recall, precision, f1, mean_accuracy, std_deviation)
 
        grafico_nuvem_de_palavras_negativa_positiva(df_predicted)
    
    else:
        st.write(f'''<p style='text-align: center'>
                 <br>Utilize as caixas de seleção acima para selecionar os dados para análise de sentimento.</p>
                ''', unsafe_allow_html=True)
    
############################################################ - ############################################################
    
# Função para mostrar informações sobre o jogo
def informacoes_sobre_jogo(df):

    st.write(f'''<h3 style='text-align: center'>
            <br>Dados sobre as avaliações<br><br></h3>
            ''', unsafe_allow_html=True)

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
        
def matriz_confusao(y_test, y_pred):

    st.write(f'''<h3 style='text-align: center'>
            <br>Matriz de Confusão<br><br></h3>
            ''', unsafe_allow_html=True)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")
    sns.color_palette("flare", as_cmap=True)
    heatmap = sns.heatmap(cm, annot=True, fmt="d", annot_kws={"fontsize": 10})

    ax.set_xticklabels(["Negativas (Previsto)", "Positivas (Previsto)"], fontsize=8)
    ax.set_yticklabels(["Negativas (Real)", "Positivas (Real)"], fontsize=8)

    # Ajuste o tamanho da fonte da barra de cores
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)

    st.pyplot(fig, use_container_width=True)
        
def informacoes_classificador(acuracia, recall, precisao, f1, acuracia_crossvalidation, desvio_padrao):

    st.write(f'''<h3 style='text-align: center'>
             <br>Métricas do modelo<br><br></h3>
             ''', unsafe_allow_html=True)

    col1, col2 = st.columns([2,2], gap="small")

    with col1:
        st.write(f'''<p>Acurácia: {acuracia:.4f}</p>
                ''', unsafe_allow_html=True)
        
        st.write(f'''<p>Recall: {recall:.4f}</p>
                ''', unsafe_allow_html=True)
        
        st.write(f'''<p>Precisão: {precisao:.4f}</p>
                ''', unsafe_allow_html=True)
        
    with col2:
        st.write(f'''<p>F1-Score: {f1:.4f}</p>
                ''', unsafe_allow_html=True)
        
        st.write(f'''<p>Acurácia média utilizando cross-validation: {acuracia_crossvalidation:.4f}</p>
                ''', unsafe_allow_html=True)
        
        st.write(f'''<p>Desvio padrão de: {desvio_padrao:.4f}</p>
                ''', unsafe_allow_html=True)
    
    st.text("")
    st.text("")

# Função para gerar o gráfico de avaliações por sentimento
def grafico_avaliacoes_sentimento(df):
    st.write(f'''<h3 style='text-align: center'>
        Histograma das avaliações por sentimento</h3>
        ''', unsafe_allow_html=True)
    
    fig = px.histogram(df, x="predicted_sentiment", nbins=2, labels=dict(predicted_sentiment="Polaridade prevista ", count="Contagem de avaliações "))
    
    # Verifica o número de avaliações negativas e positivas
    num_negativas = len(df[df['predicted_sentiment'] == 0])
    num_positivas = len(df[df['predicted_sentiment'] == 1])

    # Define as cores com base nas condições
    if num_negativas > 0 and num_positivas > 0 or num_negativas > 0:
        colors = ['#FF4136', '#2ECC40']
    else:
        colors = ['#2ECC40']
    fig.update_traces(marker_color=colors)
    fig.update_xaxes(ticktext=[" Negativas", " Positivas"],
                     tickvals=[0,1])
    fig.update_yaxes(title_text = "Contagem de Avaliações")

    # Ajuste o espaçamento entre as colunas (bargap) e os limites do eixo x (range_x)
    fig.update_layout(bargap=0.2, xaxis_range=[-0.5, 1.5])

    st.plotly_chart(fig)

def grafico_nuvem_de_palavras_negativa_positiva(df):
    #Criar o wordcloud pra reviews positivas e negativas
    positive_text = " ".join(df[df["predicted_sentiment"] == 1]["review_text"])
    negative_text = " ".join(df[df["predicted_sentiment"] == 0]["review_text"])
    
    # Verificar se o texto não está vazio antes de criar a nuvem de palavras
    if len(positive_text) > 0:
        positive_wordcloud = WordCloud(width=800, height=400, background_color="black", max_words=25).generate_from_text(positive_text)
        st.write(f'''<h3 style='text-align: center'>
                 Nuvem de palavras de avaliações positivas<br><br></h3>
                 ''', unsafe_allow_html=True)
        st.image(positive_wordcloud.to_image())
        st.write("")
    else:
        st.write(f'''<h3 style='text-align: center'>
                Nuvem de palavras de avaliações positivas<br><br></h3>
                ''', unsafe_allow_html=True)
        st.write(f'''<p style='text-align: center'>
                Nenhuma avaliação positiva encontrada para este jogo.<br></p>
                ''', unsafe_allow_html=True)
        st.write("")
    
    if len(negative_text) > 0:
        negative_wordcloud = WordCloud(width=800, height=400, background_color="black", max_words=25).generate_from_text(negative_text)
        st.write(f'''<h3 style='text-align: center'>
                 Nuvem de palavras de avaliações negativas<br><br></h3>
                 ''', unsafe_allow_html=True)
        st.image(negative_wordcloud.to_image())
        st.write("")
    else:
        st.write(f'''<h3 style='text-align: center'>
                Nuvem de palavras de avaliações negativas<br><br></h3>
                ''', unsafe_allow_html=True)
        st.write(f'''<p style='text-align: center'>
                Nenhuma avaliação negativa encontrada para este jogo.<br></p>
                ''', unsafe_allow_html=True)
        st.write("")

def main():
    build_header()
    build_body()

if __name__ == "__main__":
    main()

