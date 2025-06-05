import streamlit as st
import hashlib
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
from langchain.schema import Document

# Configuração inicial
st.set_page_config(page_title="Chat com Arquivos", layout="wide")

# Carregar variáveis de ambiente
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
TESSERACT_PATH = os.environ.get("TESSERACT_PATH") or st.secrets.get("TESSERACT_PATH", "/usr/bin/tesseract")

# Configurar API Key do OpenAI
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Configurar Tesseract se disponível
try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
except:
    pass

# Sistema de autenticação simples
def check_password():
    """Retorna True se o usuário inseriu a senha correta."""
    
    def password_entered():
        """Verifica se a senha inserida está correta."""
        if (st.session_state["username"] == "Hisoka" and 
            st.session_state["password"] == "Hisoka123#"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Remove a senha da sessão
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Primeira execução ou logout
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    # Se ainda não está autenticado
    if not st.session_state["password_correct"]:
        # Formulário de login
        st.title("🔐 Login")
        with st.form("login_form"):
            st.text_input("Usuário", key="username")
            st.text_input("Senha", type="password", key="password")
            st.form_submit_button("Entrar", on_click=password_entered)
            
        if "password_correct" in st.session_state and st.session_state["password_correct"] == False:
            st.error("😕 Usuário ou senha incorretos")
            
        return False
    
    return True

# Função de logout
def logout():
    st.session_state["password_correct"] = False
    st.rerun()

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
    
    # Criar arquivo temporário
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
            except Exception:
                st.warning(f"⚠️ OCR não disponível para {file.name}")
    except Exception as e:
        st.error(f"❌ Erro em {file.name}: {str(e)}")
    finally:
        # Limpar arquivo temporário
        try:
            os.remove(tmp_path)
        except:
            pass
    
    return documentos

def criar_vectorstore(documentos):
    """Cria vectorstore a partir dos documentos"""
    if not documentos:
        return None
    
    try:
        # Dividir documentos em chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_documents(documentos)
        
        # Criar embeddings e vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore
    except Exception as e:
        st.error(f"Erro ao criar vectorstore: {str(e)}")
        return None

# Interface lateral
with st.sidebar:
    st.header("💬 Gerenciar Sessões")
    
    # Seletor de sessões
    if st.session_state.sessoes:
        sessao_selecionada = st.selectbox(
            "Sessão ativa:",
            options=list(st.session_state.sessoes.keys()),
            index=list(st.session_state.sessoes.keys()).index(st.session_state.sessao_atual)
        )
        
        if sessao_selecionada != st.session_state.sessao_atual:
            st.session_state.sessao_atual = sessao_selecionada
            st.rerun()
    
    # Botão nova sessão
    if st.button("➕ Nova Sessão", use_container_width=True):
        criar_nova_sessao()
    
    st.divider()
    
    # Upload de arquivos
    st.header("📤 Carregar Arquivos")
    uploaded_files = st.file_uploader(
        "Escolha os arquivos",
        type=["pdf", "docx", "doc", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.sessao_atual}"
    )
    
    if uploaded_files:
        if st.button("🔄 Processar Arquivos", use_container_width=True):
            with st.spinner("Processando..."):
                todos_docs = []
                
                # Processar cada arquivo
                for file in uploaded_files:
                    docs = processar_arquivo(file)
                    todos_docs.extend(docs)
                
                if todos_docs:
                    # Criar vectorstore
                    vectorstore = criar_vectorstore(todos_docs)
                    if vectorstore:
                        st.session_state.sessoes[st.session_state.sessao_atual]["vectorstore"] = vectorstore
                        st.success(f"✅ {len(todos_docs)} documentos processados!")
                else:
                    st.error("❌ Nenhum documento foi processado")
    
    # Status e reset
    st.divider()
    vectorstore_atual = st.session_state.sessoes[st.session_state.sessao_atual].get("vectorstore")
    
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

# Mostrar histórico
historico = st.session_state.sessoes[st.session_state.sessao_atual]["historico"]
for pergunta, resposta in historico:
    with st.chat_message("user"):
        st.write(pergunta)
    with st.chat_message("assistant"):
        st.write(resposta)

# Input de pergunta
pergunta = st.chat_input("Digite sua pergunta...")

if pergunta:
    # Adicionar pergunta ao chat
    with st.chat_message("user"):
        st.write(pergunta)
    
    # Processar resposta
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                vectorstore = st.session_state.sessoes[st.session_state.sessao_atual].get("vectorstore")
                
                if vectorstore:
                    # Chat com documentos
                    retriever = vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    )
                    
                    # Criar chain
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
                    
                    # Obter resposta
                    resposta = chain({"question": pergunta})["answer"]
                else:
                    # Chat sem documentos
                    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
                    resposta = llm.predict(pergunta)
                
                # Mostrar resposta
                st.write(resposta)
                
                # Adicionar ao histórico
                historico.append((pergunta, resposta))
                
            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")
