import streamlit as st
 
st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
#sidebar
 
with st.sidebar:
    st.markdown("## DocMind")
    st.markdown("Votre assistant RAG intelligent")
 
#main
 
st.title(" DocMind")
st.write("Votre assistant RAG intelligent ")