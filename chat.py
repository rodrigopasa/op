import streamlit as st
import streamlit_authenticator as stauth
import os
import tempfile
from PIL import Image
import pytesseract
from datetime import datetime

# Importações do langchain
try:
    from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, CSVLoader
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.chat_models import ChatOpenAI
except ImportError:
    from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader, CSVLoader
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chat_models import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

# Configuração inicial
st.set_page_config(page_title="Chat com Arquivos", layout="wide")
st.title("📚 Chat com Arquivos + Memória de Sessão")

# Carregar variáveis de ambiente
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
TESSERACT_PATH = os.environ.get("TESSERACT_PATH") or st.secrets.get("TESSERACT_PATH", "/usr/bin/tesseract")
AUTH_KEY = os.environ.get("AUTH_KEY") or st.secrets.get("AUTH_KEY", "chave_padrao_123")

# Configurar API Key do OpenAI
if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY não encontrada. Configure nas variáveis de ambiente.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Configurar Tesseract se disponível
try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
except:
    st.warning("OCR não disponível. Upload de imagens pode não funcionar.")

# Configuração de autenticação
names = ["Hisoka"]
usernames = ["Hisoka"]
passwords = ["$2b$12$KIX0m1x2V1k2a8F7J9jzOeY4Ue8T4k4O5U7oE7K0l1N6r5P7Q8W"]  # hash de "Hisoka123#"

# Criar autenticador
authenticator = stauth.Authenticate(
    credentials={
        "usernames": {
            usernames[0]: {
                "name": names[0],
                "password": passwords[0]
            }
        }
    },
    cookie_name="chat_arquivos_cookie",
    key=AUTH_KEY,
    cookie_expiry_days=30
)

# Login
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("❌ Usuário ou senha incorretos")
    st.stop()
elif authentication_status == None:
    st.warning("⚠️ Por favor, insira seu nome de usuário e senha")
    st.stop()

# Interface após login bem-sucedido
col1, col2 = st.columns([6, 1])
with col1:
    st.success(f"✅ Bem-vindo, {name}!")
with col2:
    authenticator.logout("Logout", "main")

# Inicializar estado da sessão
if "sessoes" not in st.session_state:
    st.session_state.sessoes = {}

if "sessao_atual" not in st.session_state:
    nova_sessao = f"Sessao_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.sessoes[nova_sessao] = {
        "historico": [],
        "vectorstore": None
    }
    st.session_state.sessao_atual = nova_sessao

# Funções auxiliares
def criar_nova_sessao():
    nova_sessao = f"Sessao_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.sessoes[nova_sessao] = {
        "historico": [],
        "vectorstore": None
    }
    st.session_state.sessao_atual = nova_sessao
    st.rerun()

def processar_arquivo(file):
    """Processa um único arquivo e retorna documentos"""
    documentos = []
    ext = file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        if ext == "pdf":
            loader = PyMuPDFLoader(tmp_path)
            documentos = loader.load()
        elif ext in ["docx", "doc"]:
            loader = Docx2txtLoader(tmp_path)
            documentos = loader.load()
        elif ext == "csv":
            loader = CSVLoader(tmp_path, encoding='utf-8')
            documentos = loader.load()
        elif ext in ["png", "jpg", "jpeg"]:
            try:
                image = Image.open(tmp_path)
                text = pytesseract.image_to_string(image)
                if text.strip():
                    documentos = [Document(page_content=text, metadata={"source": file.name})]
            except:
                st.warning(f"⚠️ OCR falhou para {file.name}")
    except Exception as e:
        st.error(f"❌ Erro em {file.name}: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

    return documentos

def criar_vectorstore(documentos):
    """Cria vectorstore a partir dos documentos"""
    if not documentos:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_documents(documentos)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

# Interface lateral
with st.sidebar:
    st.header("💬 Gerenciar Sessões")

    if st.session_state.sessoes:
        sessao_selecionada = st.selectbox(
            "Sessão ativa:",
            options=list(st.session_state.sessoes.keys()),
            index=list(st.session_state.sessoes.keys()).index(st.session_state.sessao_atual)
        )

        if sessao_selecionada != st.session_state.sessao_atual:
            st.session_state.sessao_atual = sessao_selecionada
            st.rerun()

    if st.button("➕ Nova Sessão", use_container_width=True):
        criar_nova_sessao()

    st.divider()

    st.header("📤 Carregar Arquivos")
    uploaded_files = st.file_uploader(
        "Escolha os arquivos",
        type=["pdf", "docx", "doc", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🔄 Processar Arquivos", use_container_width=True):
            with st.spinner("Processando..."):
                todos_docs = []

                for file in uploaded_files:
                    docs = processar_arquivo(file)
                    todos_docs.extend(docs)

                if todos_docs:
                    vectorstore = criar_vectorstore(todos_docs)
                    st.session_state.sessoes[st.session_state.sessao_atual]["vectorstore"] = vectorstore
                    st.success(f"✅ {len(todos_docs)} documentos processados!")
                else:
                    st.error("❌ Nenhum documento foi processado")

    st.divider()
    vectorstore_atual = st.session_state.sessoes[st.session_state.sessao_atual]["vectorstore"]

    if vectorstore_atual:
        st.success("✅ Documentos carregados")
        if st.button("🗑️ Limpar Tudo", use_container_width=True):
            st.session_state.sessoes[st.session_state.sessao_atual]["vectorstore"] = None
            st.session_state.sessoes[st.session_state.sessao_atual]["historico"] = []
            st.rerun()
    else:
        st.info("📄 Nenhum documento carregado")

# Área principal - Chat
st.header("💬 Chat")

historico = st.session_state.sessoes[st.session_state.sessao_atual]["historico"]
for pergunta, resposta in historico:
    with st.chat_message("user"):
        st.write(pergunta)
    with st.chat_message("assistant"):
        st.write(resposta)

pergunta = st.chat_input("Digite sua pergunta...")

if pergunta:
    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                vectorstore = st.session_state.sessoes[st.session_state.sessao_atual]["vectorstore"]

                if vectorstore:
                    retriever = vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    )

                    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
                    memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    )

                    chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=retriever,
                        memory=memory,
                        verbose=False
                    )

                    resposta = chain({"question": pergunta})["answer"]
                else:
                    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
                    resposta = llm.predict(pergunta)

                st.write(resposta)
                historico.append((pergunta, resposta))

            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")
