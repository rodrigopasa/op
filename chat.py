import streamlit as st
import os
import tempfile
from PIL import Image
import pytesseract
from datetime import datetime
import psycopg2
from psycopg2 import sql

# Importações do langchain (tenta importar as versões da comunidade, senão as originais)
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
from langchain.schema import Document

# Configuração inicial
st.set_page_config(page_title="Chat com Arquivos", layout="wide")

# Carregar variáveis de ambiente
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
TESSERACT_PATH = os.environ.get("TESSERACT_PATH") or st.secrets.get("TESSERACT_PATH", "")
DB_CONNECTION = os.environ.get("DATABASE_URL") or st.secrets.get("DATABASE_URL", "")

# Configurar API Key do OpenAI
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Configurar Tesseract se disponível
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Função de conexão com o banco de dados
def get_db_connection():
    return psycopg2.connect(DB_CONNECTION)

# Função para inserir arquivos no banco de dados
def insert_file_to_db(filename, filedata):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            sql.SQL("INSERT INTO file_storage (filename, filedata) VALUES (%s, %s)"),
            (filename, psycopg2.Binary(filedata))
        )
        conn.commit()
    except Exception as e:
        st.error(f"Erro ao inserir arquivo no banco: {e}")
    finally:
        cur.close()
        conn.close()

# Sistema de autenticação simples
def check_password():
    """Retorna True se o usuário inseriu a senha correta."""
    def password_entered():
        if (st.session_state.get("username") == "Hisoka" and 
            st.session_state.get("password") == "Hisoka123#"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.title("🔐 Login")
        with st.form("login_form"):
            st.text_input("Usuário", key="username")
            st.text_input("Senha", type="password", key="password")
            st.form_submit_button("Entrar", on_click=password_entered)
        if st.session_state.get("password_correct") is False:
            st.error("😕 Usuário ou senha incorretos")
        return False
    return True

# Função de logout
def logout():
    st.session_state["password_correct"] = False
    st.experimental_rerun()

# Verificar autenticação
if not check_password():
    st.stop()

# Interface principal após login
st.title("📚 Chat com Arquivos + Memória de Sessão")

# Botão de logout no canto superior direito
col1, col2 = st.columns([6, 1])
with col1:
    st.success("✅ Bem-vindo, Hisoka!")
with col2:
    if st.button("🚪 Sair"):
        logout()

# Verificar se a API Key está configurada
if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY não encontrada. Configure nas variáveis de ambiente.")
    st.stop()

# Inicializar estado da sessão para sessões de chat
if "sessoes" not in st.session_state:
    st.session_state["sessoes"] = {}

if "sessao_atual" not in st.session_state:
    nova_sessao = f"Sessao_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state["sessoes"][nova_sessao] = {
        "historico": [],
        "vectorstore": None
    }
    st.session_state["sessao_atual"] = nova_sessao

def criar_nova_sessao():
    nova_sessao = f"Sessao_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state["sessoes"][nova_sessao] = {
        "historico": [],
        "vectorstore": None
    }
    st.session_state["sessao_atual"] = nova_sessao
    st.experimental_rerun()

def processar_arquivo(file):
    """Processa um único arquivo e retorna documentos."""
    documentos = []
    ext = file.name.split(".")[-1].lower()

    # Criar arquivo temporário para leitura
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(file.getbuffer())
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
            except Exception:
                st.warning(f"⚠️ OCR não disponível para {file.name}")
    except Exception as e:
        st.error(f"❌ Erro ao processar {file.name}: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return documentos

def criar_vectorstore(documentos):
    """Cria vectorstore a partir dos documentos."""
    if not documentos:
        return None
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_documents(documentos)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Erro ao criar vectorstore: {str(e)}")
        return None

# Sidebar: gerenciamento de sessões e upload
with st.sidebar:
    st.header("💬 Gerenciar Sessões")
    sessoes_keys = list(st.session_state["sessoes"].keys())
    sessao_atual = st.session_state["sessao_atual"]

    if sessoes_keys:
        sessao_selecionada = st.selectbox(
            "Sessão ativa:",
            options=sessoes_keys,
            index=sessoes_keys.index(sessao_atual)
        )
        if sessao_selecionada != sessao_atual:
            st.session_state["sessao_atual"] = sessao_selecionada
            st.experimental_rerun()

    if st.button("➕ Nova Sessão", use_container_width=True):
        criar_nova_sessao()

    st.divider()
    st.header("📤 Carregar Arquivos")
    uploaded_files = st.file_uploader(
        "Escolha os arquivos",
        type=["pdf", "docx", "doc", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=f"uploader_{sessao_atual}"
    )

    if uploaded_files:
        if st.button("🔄 Processar Arquivos", use_container_width=True):
            with st.spinner("Processando..."):
                todos_docs = []
                for file in uploaded_files:
                    # Como file.read() esgota o buffer, usamos file.getbuffer() para armazenar e processar
                    try:
                        file_bytes = file.getbuffer()
                        insert_file_to_db(file.name, file_bytes)
                    except Exception as e:
                        st.warning(f"Falha ao salvar {file.name} no banco: {e}")

                    docs = processar_arquivo(file)
                    todos_docs.extend(docs)

                if todos_docs:
                    vectorstore = criar_vectorstore(todos_docs)
                    if vectorstore:
                        st.session_state["sessoes"][sessao_atual]["vectorstore"] = vectorstore
                        st.success(f"✅ {len(todos_docs)} documentos processados!")
                else:
                    st.error("❌ Nenhum documento foi processado")

    st.divider()
    vectorstore_atual = st.session_state["sessoes"][sessao_atual].get("vectorstore")
    if vectorstore_atual:
        st.success("✅ Documentos carregados")
        if st.button("🗑️ Limpar Tudo", use_container_width=True):
            st.session_state["sessoes"][sessao_atual]["vectorstore"] = None
            st.session_state["sessoes"][sessao_atual]["historico"] = []
            st.experimental_rerun()
    else:
        st.info("📄 Nenhum documento carregado")

# Área principal: chat
st.header("💬 Chat")

historico = st.session_state["sessoes"][st.session_state["sessao_atual"]]["historico"]

# Exibir histórico
for pergunta, resposta in historico:
    with st.chat_message("user"):
        st.write(pergunta)
    with st.chat_message("assistant"):
        st.write(resposta)

# Input de pergunta
pergunta = st.chat_input("Digite sua pergunta...")

if pergunta:
    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                vectorstore = st.session_state["sessoes"][st.session_state["sessao_atual"]].get("vectorstore")

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
                st.error(f"Erro ao gerar resposta: {str(e)}")
