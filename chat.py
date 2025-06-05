import streamlit as st
import os
import tempfile
from PIL import Image
import pytesseract
from datetime import datetime
import psycopg2
from psycopg2 import sql
import pickle
import base64

# Importações do langchain
try:
    from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, CSVLoader
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader, CSVLoader
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Configuração da página com tema escuro e layout wide
st.set_page_config(
    page_title="🤖 AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhorar o visual
st.markdown("""
<style>
/* Tema principal */
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.chat-container {
    background: #f8f9fa;
    border-radius: 15px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.sidebar-section {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.success-box {
    background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.warning-box {
    background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.file-item {
    background: #e9ecef;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #667eea;
}

/* Melhorar aparência dos botões */
.stButton > button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

/* Melhorar chat messages */
.stChatMessage {
    background: white;
    border-radius: 15px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

/* Login form styling */
.login-container {
    max-width: 400px;
    margin: 0 auto;
    padding: 2rem;
    background: white;
    border-radius: 15px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
}

.upload-zone {
    border: 2px dashed #667eea;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    background: #f8f9ff;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Carregar variáveis de ambiente
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
TESSERACT_PATH = os.environ.get("TESSERACT_PATH") or st.secrets.get("TESSERACT_PATH", "")
DB_CONNECTION = os.environ.get("DATABASE_URL") or st.secrets.get("DATABASE_URL", "")

# Configurar API Key do OpenAI
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Configurar Tesseract se disponível
if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Função de conexão com o banco de dados
def get_db_connection():
    if not DB_CONNECTION:
        return None
    try:
        return psycopg2.connect(DB_CONNECTION)
    except Exception as e:
        st.error(f"Erro na conexão com banco de dados: {e}")
        return None

# Função para criar tabelas
def create_tables():
    if not DB_CONNECTION:
        return
    
    commands = [
        """
        CREATE TABLE IF NOT EXISTS file_storage (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            filedata BYTEA NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size INTEGER,
            file_type VARCHAR(50)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) UNIQUE NOT NULL,
            historico JSONB,
            vectorstore_data BYTEA,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    ]
    
    conn = get_db_connection()
    if not conn:
        return
        
    try:
        cur = conn.cursor()
        for command in commands:
            cur.execute(command)
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        st.error(f"❌ Erro ao criar tabelas: {error}")
    finally:
        conn.close()

# Criar tabelas na inicialização
create_tables()

# Função para salvar vectorstore no banco
def save_vectorstore_to_db(session_id, vectorstore):
    if not DB_CONNECTION or not vectorstore:
        return
    
    conn = get_db_connection()
    if not conn:
        return
        
    try:
        # Serializar vectorstore
        vectorstore_bytes = pickle.dumps(vectorstore)
        
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO chat_sessions (session_id, vectorstore_data) 
            VALUES (%s, %s)
            ON CONFLICT (session_id) 
            DO UPDATE SET vectorstore_data = EXCLUDED.vectorstore_data, updated_at = CURRENT_TIMESTAMP
        """, (session_id, psycopg2.Binary(vectorstore_bytes)))
        conn.commit()
        cur.close()
    except Exception as e:
        st.error(f"Erro ao salvar vectorstore: {e}")
    finally:
        conn.close()

# Função para carregar vectorstore do banco
def load_vectorstore_from_db(session_id):
    if not DB_CONNECTION:
        return None
        
    conn = get_db_connection()
    if not conn:
        return None
        
    try:
        cur = conn.cursor()
        cur.execute("SELECT vectorstore_data FROM chat_sessions WHERE session_id = %s", (session_id,))
        result = cur.fetchone()
        cur.close()
        
        if result and result[0]:
            return pickle.loads(result[0])
    except Exception as e:
        st.error(f"Erro ao carregar vectorstore: {e}")
    finally:
        conn.close()
    
    return None

# Função para inserir arquivos no banco de dados
def insert_file_to_db(filename, filedata, file_size, file_type):
    if not DB_CONNECTION:
        return
        
    conn = get_db_connection()
    if not conn:
        return
        
    try:
        cur = conn.cursor()
        cur.execute(
            sql.SQL("INSERT INTO file_storage (filename, filedata, file_size, file_type) VALUES (%s, %s, %s, %s)"),
            (filename, psycopg2.Binary(filedata), file_size, file_type)
        )
        conn.commit()
        cur.close()
    except Exception as e:
        st.error(f"❌ Erro ao inserir arquivo no banco: {e}")
    finally:
        conn.close()

# Função para recuperar arquivos do banco de dados
def get_files_from_db():
    if not DB_CONNECTION:
        return []
        
    conn = get_db_connection()
    if not conn:
        return []
        
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, filename, upload_date, file_size, file_type 
            FROM file_storage 
            ORDER BY upload_date DESC
        """)
        files = cur.fetchall()
        cur.close()
        return files
    except Exception as e:
        st.error(f"❌ Erro ao recuperar arquivos do banco: {e}")
        return []
    finally:
        conn.close()

# Função de autenticação
def check_password():
    """Retorna True se o usuário inseriu a senha correta."""
    def password_entered():
        if (st.session_state.get("username", "").strip() == "admin" and 
            st.session_state.get("password", "") == "admin123"):
            st.session_state["password_correct"] = True
            # Limpar campos sensíveis
            if "password" in st.session_state:
                del st.session_state["password"]
            if "username" in st.session_state:
                del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        # Container centralizado para login
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.markdown("# 🔐 Login")
            st.markdown("---")
            
            with st.form("login_form"):
                st.text_input("👤 Usuário", key="username", placeholder="Digite seu usuário")
                st.text_input("🔒 Senha", type="password", key="password", placeholder="Digite sua senha")
                submitted = st.form_submit_button("🚀 Entrar", use_container_width=True)
                
                if submitted:
                    password_entered()
            
            if st.session_state.get("password_correct") is False:
                st.markdown('<div class="warning-box">😕 Usuário ou senha incorretos</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        return False
    return True

# Função de logout
def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Verificar autenticação
if not check_password():
    st.stop()

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Chat Assistant</h1>
    <p>Converse com seus documentos usando Inteligência Artificial</p>
</div>
""", unsafe_allow_html=True)

# Verificar se a API Key está configurada
if not OPENAI_API_KEY:
    st.markdown('<div class="warning-box">⚠️ OPENAI_API_KEY não encontrada. Configure nas variáveis de ambiente.</div>', unsafe_allow_html=True)
    st.stop()

# Botão de logout no header
col1, col2 = st.columns([8, 1])
with col1:
    st.markdown('<div class="success-box">✅ Bem-vindo, admin!</div>', unsafe_allow_html=True)
with col2:
    if st.button("🚪 Sair", use_container_width=True):
        logout()

# Inicializar estado da sessão
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
    st.rerun()

def formatar_tamanho_arquivo(bytes):
    """Converte bytes para formato legível"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"

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
                else:
                    st.warning(f"⚠️ Nenhum texto encontrado na imagem {file.name}")
            except Exception as e:
                st.warning(f"⚠️ OCR não disponível para {file.name}: {str(e)}")
        else:
            st.error(f"❌ Tipo de arquivo não suportado: {ext}")
            
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
        
        if not chunks:
            st.warning("⚠️ Nenhum conteúdo encontrado nos documentos")
            return None
            
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"❌ Erro ao criar vectorstore: {str(e)}")
        return None

# Sidebar: gerenciamento de sessões e upload
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 💬 Gerenciar Sessões")
    
    sessoes_keys = list(st.session_state["sessoes"].keys())
    sessao_atual = st.session_state["sessao_atual"]

    if sessoes_keys:
        sessao_selecionada = st.selectbox(
            "🔄 Sessão ativa:",
            options=sessoes_keys,
            index=sessoes_keys.index(sessao_atual) if sessao_atual in sessoes_keys else 0,
            format_func=lambda x: f"📝 {x.replace('Sessao_', '').replace('_', ' ')}"
        )
        if sessao_selecionada != sessao_atual:
            st.session_state["sessao_atual"] = sessao_selecionada
            st.rerun()

    if st.button("➕ Nova Sessão", use_container_width=True):
        criar_nova_sessao()
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Upload de arquivos
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Carregar Arquivos")
    
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "📁 Escolha os arquivos",
        type=["pdf", "docx", "doc", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=f"uploader_{sessao_atual}",
        help="Suporte: PDF, Word, Excel, CSV, Imagens"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} arquivo(s) selecionado(s)")
        
        if st.button("🔄 Processar Arquivos", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            todos_docs = []
            total_files = len(uploaded_files)
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processando: {file.name}")
                progress_bar.progress((i + 1) / total_files)
                
                try:
                    file_bytes = file.getbuffer()
                    file_size = len(file_bytes)
                    file_type = file.name.split(".")[-1].lower()
                    
                    # Salvar no banco de dados
                    if DB_CONNECTION:
                        insert_file_to_db(file.name, file_bytes, file_size, file_type)

                    # Processar documento
                    docs = processar_arquivo(file)
                    todos_docs.extend(docs)
                    
                except Exception as e:
                    st.error(f"❌ Erro ao processar {file.name}: {e}")

            if todos_docs:
                status_text.text("Criando índice de busca...")
                vectorstore = criar_vectorstore(todos_docs)
                if vectorstore:
                    st.session_state["sessoes"][sessao_atual]["vectorstore"] = vectorstore
                    # Salvar vectorstore no banco
                    if DB_CONNECTION:
                        save_vectorstore_to_db(sessao_atual, vectorstore)
                    
                    progress_bar.progress(1.0)
                    status_text.text("")
                    st.success(f"🎉 {len(todos_docs)} documentos processados com sucesso!")
                else:
                    st.error("❌ Falha ao criar índice de busca")
            else:
                st.error("❌ Nenhum documento foi processado")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Lista de arquivos carregados
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📄 Documentos Carregados")
    
    arquivos_carregados = get_files_from_db()
    if arquivos_carregados:
        st.success(f"📊 Total: {len(arquivos_carregados)} arquivo(s)")
        
        with st.expander("Ver detalhes", expanded=False):
            for file_id, filename, upload_date, file_size, file_type in arquivos_carregados:
                size_str = formatar_tamanho_arquivo(file_size) if file_size else "N/A"
                st.markdown(f"""
                <div class="file-item">
                    <strong>📎 {filename}</strong><br>
                    <small>📅 {upload_date.strftime('%d/%m/%Y %H:%M')}</small><br>
                    <small>📏 {size_str} • 🏷️ {file_type.upper()}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📭 Nenhum documento carregado ainda")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Área principal: chat
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown("## 💬 Conversa")

# Carregar vectorstore do banco se não estiver na sessão
sessao_data = st.session_state["sessoes"][st.session_state["sessao_atual"]]
if not sessao_data.get("vectorstore") and DB_CONNECTION:
    vectorstore_db = load_vectorstore_from_db(st.session_state["sessao_atual"])
    if vectorstore_db:
        sessao_data["vectorstore"] = vectorstore_db

historico = sessao_data["historico"]

# Exibir histórico de chat
if historico:
    for pergunta, resposta in historico:
        with st.chat_message("user", avatar="👤"):
            st.write(pergunta)
        with st.chat_message("assistant", avatar="🤖"):
            st.write(resposta)
else:
    st.info("👋 Olá! Faça upload de documentos e comece a conversar comigo sobre eles!")

# Input de pergunta
pergunta = st.chat_input("💭 Digite sua pergunta aqui...")

if pergunta:
    with st.chat_message("user", avatar="👤"):
        st.write(pergunta)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🧠 Pensando..."):
            try:
                vectorstore = sessao_data.get("vectorstore")

                if vectorstore:
                    # Chat com contexto dos documentos
                    retriever = vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    )
                    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4")
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
                    resultado = chain({"question": pergunta})
                    resposta = resultado["answer"]
                else:
                    # Chat sem contexto
                    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4")
                    resposta_obj = llm.invoke(pergunta)
                    resposta = resposta_obj.content if hasattr(resposta_obj, 'content') else str(resposta_obj)

                st.write(resposta)
                historico.append((pergunta, resposta))
                
            except Exception as e:
                error_msg = f"❌ Erro ao gerar resposta: {str(e)}"
                st.error(error_msg)
                historico.append((pergunta, error_msg))

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Sessões Ativas", len(st.session_state["sessoes"]))
with col2:
    st.metric("💬 Mensagens", len(historico))
with col3:
    vectorstore_status = "✅ Ativo" if sessao_data.get("vectorstore") else "❌ Inativo"
    st.metric("🔍 Busca Inteligente", vectorstore_status)
