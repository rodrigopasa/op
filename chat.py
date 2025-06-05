import streamlit as st
import streamlit_authenticator as stauth
import psycopg2
import os
import tempfile
from PIL import Image
import pytesseract
from datetime import datetime

# Importações atualizadas de langchain_community
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

# Configuração inicial
st.set_page_config(page_title="Chat com Arquivos", layout="wide")
st.title("📚 Chat com Arquivos + Memória de Sessão")

# Carregar variáveis secrets
DATABASE_URL = st.secrets.get("DATABASE_URL")
API_KEY = st.secrets.get("OPENAI_API_KEY")
TESSERACT_PATH = st.secrets.get("TESSERACT_PATH")

# Configurar o caminho do Tesseract
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    st.warning("Caminho do Tesseract não definido. Verifique suas Secrets.")

# Autenticação com assinatura correta e sem duplicidade de argumentos
authenticator = stauth.Authenticate(
    ["Hisoka"],  # nomes
    ["Hisoka"],  # usernames
    ["$2b$12$KIX0m1x2V1k2a8F7J9jzOeY4Ue8T4k4O5U7oE7K0l1N6r5P7Q8W"],  # hash da senha
    "cookie_name",  # nome do cookie
    "signature_key",  # sua chave de assinatura
    cookie_expiry_days=30  # duração do cookie (apenas uma vez)
)

# Login
name, authentication_status, username = authenticator.login("Login", "main")
if not authentication_status:
    st.write("Usuário ou senha incorretos")
    st.stop()

# Aqui começa o restante do seu código, que só será executado após login
# ---------------------------------------------

# Inicializar sessões
if "sessoes" not in st.session_state:
    st.session_state.sessoes = {}
if "sessao_atual" not in st.session_state:
    nova_sessao = f"Sessao_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.sessoes[nova_sessao] = {"historico": [], "vectorstore": None}
    st.session_state.sessao_atual = nova_sessao

# Funções
def criar_nova_sessao():
    nova_sessao = f"Sessao_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.sessoes[nova_sessao] = {"historico": [], "vectorstore": None}
    st.session_state.sessao_atual = nova_sessao
    st.experimental_rerun()

def carregar_arquivos(uploaded_files):
    documentos = []
    for file in uploaded_files:
        ext = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if ext == "pdf":
            loader = PyMuPDFLoader(tmp_path)
            documentos.extend(loader.load())
        elif ext == "docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(tmp_path)
            documentos.extend(loader.load())
        elif ext == "csv":
            loader = CSVLoader(tmp_path)
            documentos.extend(loader.load())
        elif ext in ["png", "jpg", "jpeg"]:
            image = Image.open(tmp_path)
            text = pytesseract.image_to_string(image)
            documentos.append(Document(page_content=text))
        os.remove(tmp_path)
    return documentos

def criar_vectorstore(documentos):
    if not documentos:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_divididos = splitter.split_documents(documentos)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs_divididos, embeddings)

def mostrar_documentos(vectorstore):
    if vectorstore:
        st.sidebar.markdown("### 📄 Documentos carregados")
        docs = vectorstore.as_retriever().index.documents[:5]
        for i, doc in enumerate(docs):
            st.sidebar.markdown(f"**Doc {i+1}:** {doc.page_content[:100]}...")

# Interface de sessões
st.sidebar.header("💬 Sessões")
sessao_nome = st.sidebar.selectbox(
    "Escolha uma sessão:",
    list(st.session_state.sessoes.keys()),
    index=list(st.session_state.sessoes.keys()).index(st.session_state.sessao_atual)
)
if sessao_nome != st.session_state.sessao_atual:
    st.session_state.sessao_atual = sessao_nome

if st.sidebar.button("➕ Nova Sessão"):
    criar_nova_sessao()

# Upload de arquivos
uploaded_files = st.sidebar.file_uploader(
    "📤 Envie arquivos", type=["pdf", "docx", "csv", "png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processando arquivos..."):
        documentos = carregar_arquivos(uploaded_files)
        vectorstore = criar_vectorstore(documentos)
        st.session_state.sessoes[st.session_state.sessao_atual]["vectorstore"] = vectorstore
        st.success(f"{len(documentos)} arquivos processados!")

# Resetar documentos
if st.sidebar.button("🗑️ Resetar Documentos"):
    st.session_state.sessoes[st.session_state.sessao_atual]["vectorstore"] = None
    st.success("Vectorstore resetado.")

# Mostrar documentos carregados
vectorstore_atual = st.session_state.sessoes[st.session_state.sessao_atual]["vectorstore"]
mostrar_documentos(vectorstore_atual)

# Chat
st.markdown("---")
st.markdown("## 💬 Chat")
query = st.text_input("Digite sua pergunta:", placeholder="Ex: Qual o resumo dos documentos?")

if query:
    historico = st.session_state.sessoes[st.session_state.sessao_atual]["historico"]
    retriever = (
        vectorstore_atual.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        if vectorstore_atual else None
    )
    from langchain_memory import ConversationBufferMemory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
        retriever=retriever,
        memory=memory
    )
    resposta = chain.run(query)
    historico.append((query, resposta))
    # Mostrar histórico
    for pergunta, resposta in historico:
        with st.chat_message("user"):
            st.markdown(pergunta)
        with st.chat_message("assistant"):
            st.markdown(resposta)
