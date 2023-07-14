import pandas as pd
import streamlit as st
import webbrowser

st.set_page_config(
    page_title="Sobre",
    page_icon="📄",
    # layout="centered",
)

st.title('Sobre')
st.markdown("---")
st.header('Projeto desenvolvido para a disciplina Projeto Interdisciplinar para Sistemas de Informação III - 2022.2')
st.subheader('Grupo GGJJLM')
st.text("")
st.write("Participantes:")
st.markdown(
"""
- Gabriel Duarte da Silva
- Gabriel Moreira de Lemos e Silva
- José Fernando de Oliveira Filho
- José Francisco de Medeiros
- Leonardo de Sousa Araújo Alcântara
- Marcos de Oliveira de Jesus
""")
st.text("")

url1 = 'https://docs.google.com/document/d/151L1pRvdYTNYcvONrVlpuCh6-HuasvvWEu3KfF5aM-4'
url2 = 'https://github.com/Redoze/PISI3-2022.2'
texto = 'Acesse também o [artigo](%s) ou o [repositório](%s).' % (url1, url2)

st.markdown(texto, unsafe_allow_html=True)
