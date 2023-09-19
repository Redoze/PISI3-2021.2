import streamlit as st
from streamlit_extras.mention import mention

st.set_page_config(
    page_title="Sobre",
    page_icon="ℹ️",
    layout="centered",
)

def build_header():

    st.write(f'''<h1 style='text-align: center'>Sobre<br></h1>
             <p style='text-align: center'>Projeto desenvolvido para a disciplina Projeto Interdisciplinar para Sistemas de Informação III - 2022.2 
             do curso de Bacharelado em Sistemas de Informação (BSI).<br><br>
             Docentes: Gabriel Alves e Maria da Conceição Moraes<br>
             Grupo: GJJ</p>
             ''', unsafe_allow_html=True)
    st.markdown("---")

def build_body():

    st.write(f'''<h3 style='text-align: center'>
             Participantes<br><br></h3>
             ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns([6,4])

    with col1:
        func_mention(" Gabriel Moreira de Lemos e Silva", 
                    "github", "https://github.com/GabrielTFV")
    with col2:
        func_mention(" José Fernando de Oliveira Filho", 
                    "github", "https://github.com/fernandooliveira7")
        
    vazio1, col1, vazio2 = st.columns([3,5,1])

    with vazio1:
        pass

    with col1:
        func_mention(" José Francisco de Medeiros", 
                    "github", "https://github.com/Redoze")

    with vazio2:
        pass

    st.markdown("---")
    st.write(f'''<h3 style='text-align: center'>
             Links<br><br></h3>
             ''', unsafe_allow_html=True)

    vazio1, col1, col2, vazio2 = st.columns([1,6,3,1])
    
    with vazio1:
        pass

    with col1:
        func_mention(" Artigo",
                    "📄", "https://docs.google.com/document/d/151L1pRvdYTNYcvONrVlpuCh6-HuasvvWEu3KfF5aM-4")

        func_mention(" Repositório", 
                    "github", "https://github.com/Redoze/PISI3-2022.2")
        
    with col2:
        func_mention(" Conjunto de dados - 1", 
            "📘", "https://www.kaggle.com/datasets/andrewmvd/steam-reviews")
        func_mention(" Conjunto de dados - 2", 
            "📙", "https://www.kaggle.com/datasets/nikdavis/steam-store-games")
        func_mention(" Conjunto de dados - 3", 
            "📗", "https://data.mendeley.com/datasets/ycy3sy3vj2/1")
        
    with vazio2:
        pass

def func_mention(label, icon, url):
    mention(
    label= label,
    icon= icon,
    url= url)

build_header()
build_body()
