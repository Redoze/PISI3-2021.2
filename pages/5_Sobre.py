import streamlit as st
from streamlit_extras.mention import mention

st.set_page_config(
    page_title="Sobre",
    page_icon="ℹ️",
    layout="centered",
)

def build_header():

    st.title('Sobre')
    st.text("")
    st.write('**Projeto desenvolvido para a disciplina Projeto Interdisciplinar para Sistemas de Informação III - 2022.2 do curso de Bacharelado em Sistemas de Informação (BSI).**')
    st.write('**Grupo: GGJJM**')
    st.text("")

def build_body():

    st.write("Participantes:")
    col1, col2 = st.columns(2)

    with col1:
        func_mention(" Gabriel Duarte da Silva", 
                    "github", "https://github.com/gabrielduaarte")
        func_mention(" Gabriel Moreira de Lemos e Silva", 
                    "github", "https://github.com/GabrielTFV")
        func_mention(" José Fernando de Oliveira Filho", 
                    "github", "https://github.com/fernandooliveira7")

    with col2:
        func_mention(" José Francisco de Medeiros", 
                    "github", "https://github.com/Redoze")
        func_mention(" Marcos de Oliveira de Jesus", 
                    "github", "https://github.com/Markie98")

    st.markdown("---")
    st.subheader('Links')
    st.text("")
    col1, col2 = st.columns(2)
    
    with col1:
        func_mention(" Artigo",
                    "📄", "https://docs.google.com/document/d/151L1pRvdYTNYcvONrVlpuCh6-HuasvvWEu3KfF5aM-4")

        func_mention(" Repositório", 
                    "github", "https://github.com/Redoze/PISI3-2022.2")
        
    with col2:
        func_mention(" Dataset - 1", 
            "📘", "https://www.kaggle.com/datasets/andrewmvd/steam-reviews")
        func_mention(" Dataset - 2", 
            "📙", "https://www.kaggle.com/datasets/nikdavis/steam-store-games")
        func_mention(" Dataset - 3", 
            "📗", "https://data.mendeley.com/datasets/ycy3sy3vj2/1")

def func_mention(label, icon, url):
    mention(
    label= label,
    icon= icon,
    url= url)

build_header()
build_body()
