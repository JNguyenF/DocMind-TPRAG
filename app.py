import streamlit as st
import pymupdf

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session : on garde la base vectorielle en mémoire entre les interactions

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Extraction

def extract_documents(uploaded_files):
    documents = []

    for file in uploaded_files:
        name = file.name.lower()

        if name.endswith(".pdf"):
            pdf = pymupdf.open(stream=file.getvalue(), filetype="pdf")
            for page_number, page in enumerate(pdf):
                page_text = page.get_text()
                if page_text.strip():
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": file.name,
                                "page": page_number + 1,
                            },
                        )
                    )

        elif name.endswith((".txt", ".md")):
            file_text = file.getvalue().decode("utf-8", errors="ignore")
            if file_text.strip():
                documents.append(
                    Document(
                        page_content=file_text,
                        metadata={"source": file.name, "page": "N/A"},
                    )
                )

    return documents


# sidebar

with st.sidebar:
    st.markdown("## DocMind")
    st.markdown("Votre assistant RAG intelligent")

    st.markdown("### Documents")

    uploaded_files = st.file_uploader(
        "Charger vos documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    index_button = st.button("Indexer les documents", use_container_width=True)

# main

st.title(" DocMind")
st.write("Votre assistant RAG intelligent")

if index_button:
    if not uploaded_files:
        st.warning("⚠️ Veuillez sélectionner au moins un document.")
    else:
        try:
            with st.spinner("Analyse et indexation des documents..."):
                documents = extract_documents(uploaded_files)

                if not documents:
                    st.error("Aucun texte n'a pu être extrait.")
                    st.stop()

                #  chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=150,
                )
                chunks = splitter.split_documents(documents)

                # embeddings 
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                )

            st.success(
                f"✅ {len(documents)} document(s) extrait(s), "
                f"{len(chunks)} chunks indexés dans la base vectorielle."
            )

        except Exception as e:
            st.error(f"❌ Erreur pendant l'indexation : {e}")

if st.session_state.vectorstore is not None:
    st.info("📚 Base vectorielle prête.")