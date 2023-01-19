import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

#Dataframe
df = pd.read_excel(
    io = 'steam.xlsx',
    engine = 'openpyxl',
    sheet_name= 'steam1',
    usecols='A:R',
    nrows=500,)

# Create text
topic1 = " ".join(cat.split()[0] for cat in df.steamspy_tags)
topic2 = " ".join(cat.split()[0] for cat in df.genres)
topic3 = " ".join(cat.split()[0] for cat in df.publisher)

topic = st.selectbox('select topic',['topic1','topic2','topic3'])

# Create and generate a word cloud image:
def create_wordcloud(topic):
    if topic == 'topic1':
        topic = topic1
    elif topic == 'topic2':
        topic = topic2
    else:
        topic = topic3

    wordcloud = WordCloud().generate(topic)
    return wordcloud

wordcloud = create_wordcloud(topic)

# Display the generated image:
fig, ax = plt.subplots(figsize = (20, 12))
ax.imshow(wordcloud)
plt.axis("off")
st.pyplot(fig)