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
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

# Configuração inicial
st.set_page_config(page_title="Chat com Arquivos", layout="wide")
st.title("📚 Chat com Arquivos + Memória de Sessão")

# Carregar variáveis secrets
DATABASE_URL = st.secrets.get("DATABASE_URL", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
TESSERACT_PATH = st.secrets.get("TESSERACT_PATH", "/usr/bin/tesseract")

# Configurar API Key do OpenAI
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    st.error("Chave API do OpenAI não encontrada. Configure nas Secrets.")
    st.stop()

# Configurar o caminho do Tesseract
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Autenticação - Corrigida
authenticator = stauth.Authenticate(
    ["Hisoka"],  # nomes
    ["Hisoka"],  # usernames
    ["$2b$12$KIX0m1x2V1k2a8F7J9jzOeY4Ue8T4k4O5U7oE7K0l1N6r5P7Q8W"],  # hash da senha
    "cookie_name",  # nome do cookie
    "signature_key",  # sua chave de assinatura
    cookie_expiry_days=30  # duração do cookie
)

# Login
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Usuário ou senha incorretos")
    st.stop()
elif authentication_status == None:
    st.warning("Por favor, insira seu nome de usuário e senha")
    st.stop()

# Mensagem de boas-vindas
st.success(f"Bem-vindo, {name}!")

# Aqui começa o restante do código após login bem-sucedido
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
    st.rerun()

def carregar_arquivos(uploaded_files):
    documentos = []
    for file in uploaded_files:
        ext = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            if ext == "pdf":
                loader = PyMuPDFLoader(tmp_path)
                documentos.extend(loader.load())
            elif ext == "docx":
                loader = Docx2txtLoader(tmp_path)
                documentos.extend(loader.load())
            elif ext == "csv":
                loader = CSVLoader(tmp_path)
                documentos.extend(loader.load())
            elif ext in ["png", "jpg", "jpeg"]:
                image = Image.open(tmp_path)
                text = pytesseract.image_to_string(image)
                documentos.append(Document(page_content=text))
        except Exception as e:
            st.error(f"Erro ao processar {file.name}: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
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
        # Método correto para acessar documentos no FAISS
        st.sidebar.markdown("Documentos processados e indexados com sucesso!")

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
        if documentos:
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
    
    if vectorstore_atual:
        retriever = vectorstore_atual.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
            retriever=retriever,
            memory=memory
        )
        resposta = chain.run(query)
    else:
        # Se não houver documentos, use apenas o ChatOpenAI
        llm = ChatOpenAI(temperature=0, model_name="gpt-4")
        resposta = llm.predict(query)
    
    historico.append((query, resposta))
    
    # Mostrar histórico
    for pergunta, resp in historico:
        with st.chat_message("user"):
            st.markdown(pergunta)
        with st.chat_message("assistant"):
            st.markdown(resp)
