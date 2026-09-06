import streamlit as st
import pymupdf

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session : on garde la base vectorielle et l'historique du chat en mémoire

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0
if "documents_count" not in st.session_state:
    st.session_state.documents_count = 0

# Thème : mode sombre / mode clair

with st.sidebar:
    dark_mode = st.toggle(" Mode sombre", value=False, key="dark_mode")

if dark_mode:
    bg = "#0d1117"
    sidebar_bg = "#111827"
    card_bg = "#161d29"
    text = "#f8fafc"
    muted = "#94a3b8"
    border = "#2b3546"
    accent = "#6554f3"
else:
    bg = "#ffffff"
    sidebar_bg = "#f8fafc"
    card_bg = "#ffffff"
    text = "#111827"
    muted = "#64748b"
    border = "#e2e8f0"
    accent = "#6554f3"

st.markdown(
    f"""
<style>
.stApp {{
    background: {bg};
    color: {text};
}}
[data-testid="stSidebar"] {{
    background: {sidebar_bg};
    border-right: 1px solid {border};
}}
[data-testid="stSidebar"] * {{
    color: {text};
}}
h1,h2,h3,h4,h5,h6,p,label {{
    color: {text};
}}
[data-testid="stChatMessage"] {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 16px;
    padding: 12px 18px;
    margin-bottom: 12px;
}}
[data-testid="stExpander"] {{
    background: {sidebar_bg};
    border: 1px solid {border};
    border-radius: 12px;
}}
[data-testid="stFileUploader"] {{
    background: {card_bg};
    border: 1px dashed {border};
    border-radius: 12px;
    padding: 10px;
}}
.stButton > button {{
    width: 100%;
    border-radius: 10px;
    border: 1px solid {border};
    background: {card_bg};
    color: {text};
    font-weight: 600;
}}
.stButton > button:hover {{
    border-color: {accent};
    color: {accent};
}}
</style>
""",
    unsafe_allow_html=True,
)

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
    st.markdown("**👤 Nguyen Van Jason Fahasoavana**")

    st.markdown("### Documents")

    uploaded_files = st.file_uploader(
        "Charger vos documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    index_button = st.button("Indexer les documents", use_container_width=True)

    st.markdown("### Configuration")

    llm_enabled = st.toggle("Activer le LLM", value=False)

    st.markdown("### Statistiques")

    st.markdown(
        f"""
-  Documents indexés : **{st.session_state.documents_count}**
-  Chunks créés : **{st.session_state.chunks_count}**
"""
    )

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

                # chunks
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
                st.session_state.chunks_count = len(chunks)
                st.session_state.documents_count = len(uploaded_files)

            st.success(
                f"✅ {len(documents)} document(s) extrait(s), "
                f"{len(chunks)} chunks indexés dans la base vectorielle."
            )
            st.rerun()

        except Exception as e:
            st.error(f"❌ Erreur pendant l'indexation : {e}")

# Affichage de l'historique du chat

st.subheader("Conversation")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie du chat

question = st.chat_input("Posez une question sur vos documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.vectorstore is None:
        response = "⚠️ Veuillez d'abord charger et indexer vos documents."

        with st.chat_message("assistant"):
            st.warning(response)

    else:
        # Recherche sémantique
        results = st.session_state.vectorstore.similarity_search(question, k=3)

        if not llm_enabled:
            # Mode recherche simple
            response = f"🔎 {len(results)} fragments pertinents trouvés."

            with st.chat_message("assistant"):
                st.markdown("**🧠 DocMind**")
                st.markdown("Voici les passages les plus pertinents trouvés dans vos documents.")

                for i, doc in enumerate(results, start=1):
                    source = doc.metadata.get("source", "Source inconnue")
                    page = doc.metadata.get("page", "N/A")

                    with st.expander(f"📄 Résultat {i} — {source} | Page {page}"):
                        st.write(doc.page_content)

        else:
            # Mode RAG complet
            context = "\n\n".join(doc.page_content for doc in results)

            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
Tu es DocMind, un assistant spécialisé dans l'analyse de documents.

Réponds exclusivement à partir du contexte fourni.
N'utilise aucune connaissance extérieure.

Si la réponse n'est pas présente dans le contexte, réponds exactement :
"Je ne trouve pas cette information dans les documents."

Réponds en français de manière claire, structurée et concise.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :
""",
            )

            prompt = prompt_template.format(context=context, question=question)

            llm = OllamaLLM(model="mistral")

            try:
                with st.spinner(" DocMind réfléchit..."):
                    response = llm.invoke(prompt)

                with st.chat_message("assistant"):
                    st.markdown("**DocMind**")
                    st.markdown(response)

                    with st.expander(f"📚 Sources utilisées ({len(results)})"):
                        for i, doc in enumerate(results, start=1):
                            source = doc.metadata.get("source", "Source inconnue")
                            page = doc.metadata.get("page", "N/A")

                            st.markdown(f"**📄 {source} — Page {page}**")
                            st.write(doc.page_content)

                            if i < len(results):
                                st.divider()

            except Exception as e:
                response = f"❌ Impossible de communiquer avec Ollama : {e}"

                with st.chat_message("assistant"):
                    st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})