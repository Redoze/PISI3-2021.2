import pandas as pd
import streamlit as st
import webbrowser

st.set_page_config(
    page_title="Sobre",
    page_icon="📄",
    layout="centered",
)

st.header('Sobre')
st.text("")
st.markdown("---")
st.text("")
st.subheader('Projeto desenvolvido para a disciplina Projeto Interdisciplinar para Sistemas de Informação III 2022.2')
st.write('Grupo: #GGJJLM')
st.text("")
st.markdown(
"""
Participantes:
- Gabriel Duarte da Silva
- Gabriel Moreira de Lemos e Silva
- José Fernando de Oliveira Filho
- José Francisco de Medeiros
- Leonardo de Sousa Araújo Alcântara
- Marcos de Oliveira de Jesus
"""
)
st.text("")
url = 'https://docs.google.com/document/d/151L1pRvdYTNYcvONrVlpuCh6-HuasvvWEu3KfF5aM-4'

st.write('Acesso para o artigo:')
if st.button('Link'):
    webbrowser.open_new_tab(url)
