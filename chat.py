import streamlit as st
import os
import tempfile
from PIL import Image
import pytesseract
from datetime import datetime
import psycopg2
from psycopg2 import sql
import bcrypt
import json
import pickle
import base64
from urllib.parse import urlparse
import logging
from typing import Optional, List, Tuple, Any
from contextlib import contextmanager

# Importações do langchain com tratamento de erro melhorado
try:
    from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, CSVLoader
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Erro ao importar LangChain: {e}")
    LANGCHAIN_AVAILABLE = False

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da página com tema moderno
st.set_page_config(
    page_title="🤖 AI Chat Assistant Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS melhorado com design mais moderno
st.markdown("""
<style>
/* Reset e base */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}
/* Variáveis CSS */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --success-gradient: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    --warning-gradient: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
    --glass-bg: rgba(255, 255, 255, 0.1);
    --glass-border: rgba(255, 255, 255, 0.2);
    --shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    --border-radius: 16px;
}
/* Header principal com glassmorphism */
.main-header {
    background: var(--primary-gradient);
    padding: 2rem;
    border-radius: var(--border-radius);
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: var(--shadow);
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M20 20c0 11.046-8.954 20-20 20s-20-8.954-20-20 8.954-20 20-20 20 8.954 20 20zm-10 0c0 5.523-4.477 10-10 10s-10-4.477-10-10 4.477-10 10-10 10 4.477 10 10z'/%3E%3C/g%3E%3C/svg%3E");
}
.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 1;
}
.main-header p {
    font-size: 1.1rem;
    opacity: 0.9;
    position: relative;
    z-index: 1;
}
/* Containers com glassmorphism */
.chat-container, .sidebar-section {
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--shadow);
}
.chat-container {
    min-height: 400px;
    background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
}
/* Caixas de status melhoradas */
.success-box, .warning-box, .info-box {
    padding: 1.2rem;
    border-radius: var(--border-radius);
    color: white;
    text-align: center;
    margin: 1rem 0;
    box-shadow: var(--shadow);
    font-weight: 600;
    position: relative;
    overflow: hidden;
}
.success-box {
    background: var(--success-gradient);
}
.warning-box {
    background: var(--warning-gradient);
}
.info-box {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}
/* Animação para as caixas */
.success-box::before, .warning-box::before, .info-box::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
    transform: rotate(-45deg);
    animation: shine 3s infinite;
}
@keyframes shine {
    0% { transform: translateX(-100%) translateY(-100%) rotate(-45deg); }
    100% { transform: translateX(100%) translateY(100%) rotate(-45deg); }
}
/* File items melhorados */
.file-item {
    background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(240,240,240,0.9) 100%);
    padding: 1rem;
    border-radius: 12px;
    margin: 0.8rem 0;
    border-left: 4px solid var(--primary-gradient);
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
.file-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}
/* Botões modernos */
.stButton > button {
    background: var(--primary-gradient) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
}
/* Upload zone melhorada */
.upload-zone {
    border: 2px dashed #667eea;
    border-radius: var(--border-radius);
    padding: 2rem;
    text-align: center;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    margin: 1rem 0;
    transition: all 0.3s ease;
}
.upload-zone:hover {
    border-color: #764ba2;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
}
/* Login container melhorado */
.login-container {
    max-width: 450px;
    margin: 2rem auto;
    padding: 2.5rem;
    background: var(--glass-bg);
    backdrop-filter: blur(15px);
    border: 1px solid var(--glass-border);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
}
/* Chat messages melhoradas */
.stChatMessage {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 15px !important;
    padding: 1.2rem !important;
    margin: 0.8rem 0 !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}
/* Metrics cards */
.metric-card {
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    text-align: center;
    box-shadow: var(--shadow);
    transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 35px rgba(31, 38, 135, 0.5);
}
/* Loading spinner personalizado */
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255,255,255,.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: spin 1s ease-in-out infinite;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
/* Scrollbar personalizada */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb {
    background: var(--primary-gradient);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #5a6fd8 0%, #6b5b95 100%);
}
</style>
""", unsafe_allow_html=True)

class DatabaseManager:
    """Gerenciador de conexões com banco de dados melhorado"""
    
    def __init__(self):
        self.connection_params = self._parse_database_url()
    
  def _parse_database_url(self) -> Optional[dict]:
    """
    Parse da URL do banco de dados com tratamento de erro melhorado.
    Se a variável de ambiente ou secret DATABASE_URL não estiver definida,
    usará o valor padrão informado.
    """
    default_db_url = ("postgres://postgres:kNL6exzv6Y3HYomX4Etgpb2fqtatWIuzKh5OYozkM9NkayOywHe1i1jfyvijgS3G"
                      "@185.173.110.61:9898/postgres")
    # Tenta obter a URL do ambiente ou secrets; se não encontrar, usa o default
    db_url = os.environ.get("DATABASE_URL") or st.secrets.get("DATABASE_URL", default_db_url)
    
    if not db_url:
        logger.warning("DATABASE_URL não encontrada")
        return None
    try:
        parsed = urlparse(db_url)
        # Extrair a porta; se não estiver definida, usar 9898
        port = parsed.port or 9898

        params = {
            'host': parsed.hostname,
            'port': int(port),
            'database': parsed.path.lstrip('/'),
            'user': parsed.username,
            'password': parsed.password
        }
        # Validar parâmetros essenciais
        if not all([params['host'], params['database'], params['user']]):
            raise ValueError("Parâmetros de conexão incompletos")
        
        logger.info(f"Conexão configurada para: {params['host']}:{params['port']}/{params['database']}")
        return params
    except Exception as e:
        logger.error(f"Erro ao fazer parse da DATABASE_URL: {e}")
        st.error(f"❌ Erro na configuração do banco: {e}")
        return None



    @contextmanager
    def get_connection(self):
        """Context manager para conexões seguras"""
        if not self.connection_params:
            yield None
            return
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Erro de conexão com PostgreSQL: {e}")
            st.error(f"❌ Erro de conexão: {e}")
            yield None
        except Exception as e:
            logger.error(f"Erro inesperado na conexão: {e}")
            st.error(f"❌ Erro inesperado: {e}")
            yield None
        finally:
            if conn:
                conn.close()

    def create_tables(self) -> bool:
        """Criar tabelas com estrutura melhorada"""
        commands = [
            """
            CREATE TABLE IF NOT EXISTS file_storage (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                filedata BYTEA NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_size INTEGER,
                file_type VARCHAR(50),
                file_hash VARCHAR(64) UNIQUE,
                created_by VARCHAR(100) DEFAULT 'system'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) UNIQUE NOT NULL,
                historico JSONB,
                vectorstore_data BYTEA,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_name VARCHAR(255),
                is_active BOOLEAN DEFAULT TRUE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_file_storage_hash ON file_storage(file_hash);
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_active ON chat_sessions(is_active, updated_at);
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username) WHERE is_active = TRUE;
            """
        ]
        with self.get_connection() as conn:
            if not conn:
                return False
            try:
                with conn.cursor() as cur:
                    for command in commands:
                        cur.execute(command)
                conn.commit()
                logger.info("Tabelas criadas/atualizadas com sucesso")
                return True
            except Exception as e:
                logger.error(f"Erro ao criar tabelas: {e}")
                st.error(f"❌ Erro ao criar tabelas: {e}")
                return False

# Inicializar gerenciador de banco
db_manager = DatabaseManager()

# Resto do seu código continua igual...
# (mantém o restante do seu código original, após essa parte)

class FileProcessor:
    """Processador de arquivos melhorado"""
    
    @staticmethod
    def format_file_size(bytes_size: int) -> str:
        """Formatar tamanho do arquivo"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"
    
    @staticmethod
    def process_file(file) -> List[Document]:
        """Processar arquivo com tratamento de erro melhorado"""
        if not LANGCHAIN_AVAILABLE:
            st.error("❌ LangChain não disponível")
            return []
            
        documents = []
        ext = file.name.split(".")[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name
        
        try:
            if ext == "pdf":
                loader = PyMuPDFLoader(tmp_path)
                documents = loader.load()
            elif ext in ["docx", "doc"]:
                loader = Docx2txtLoader(tmp_path)
                documents = loader.load()
            elif ext == "csv":
                loader = CSVLoader(tmp_path, encoding='utf-8')
                documents = loader.load()
            elif ext in ["png", "jpg", "jpeg"]:
                try:
                    image = Image.open(tmp_path)
                    text = pytesseract.image_to_string(image, lang='por+eng')
                    if text.strip():
                        documents = [Document(
                            page_content=text, 
                            metadata={"source": file.name, "type": "image_ocr"}
                        )]
                    else:
                        st.warning(f"⚠️ Nenhum texto encontrado na imagem {file.name}")
                except Exception as e:
                    st.warning(f"⚠️ OCR falhou para {file.name}: {str(e)}")
            else:
                st.error(f"❌ Tipo de arquivo não suportado: {ext}")
                
        except Exception as e:
            logger.error(f"Erro ao processar {file.name}: {e}")
            st.error(f"❌ Erro ao processar {file.name}: {str(e)}")
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass
        
        return documents

class AuthManager:
    """Gerenciador de autenticação melhorado"""
    
    @staticmethod
    def check_password() -> bool:
        """Verificação de senha melhorada"""
        def password_entered():
            username = st.session_state.get("username", "").strip()
            password = st.session_state.get("password", "")
            
            # Verificação simples (em produção, usar hash)
            if username == "Hisoka" and password == "Hisoka123#":
                st.session_state["password_correct"] = True
                st.session_state["authenticated_user"] = username
                # Limpar campos sensíveis
                for key in ["password", "username"]:
                    if key in st.session_state:
                        del st.session_state[key]
            else:
                st.session_state["password_correct"] = False

        if "password_correct" not in st.session_state:
            st.session_state["password_correct"] = False

        if not st.session_state["password_correct"]:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown('<div class="login-container">', unsafe_allow_html=True)
                st.markdown("# 🔐 AI Chat Login")
                st.markdown("### Acesso Seguro ao Sistema")
                st.markdown("---")
                
                with st.form("login_form"):
                    st.text_input(
                        "👤 Usuário", 
                        key="username", 
                        placeholder="Digite seu usuário",
                        help="Digite suas credenciais de acesso"
                    )
                    st.text_input(
                        "🔒 Senha", 
                        type="password", 
                        key="password", 
                        placeholder="Digite sua senha"
                    )
                    submitted = st.form_submit_button("🚀 Entrar", use_container_width=True)
                    
                    if submitted:
                        password_entered()
                
                if st.session_state.get("password_correct") is False:
                    st.markdown(
                        '<div class="warning-box">😕 Credenciais incorretas. Tente novamente.</div>', 
                        unsafe_allow_html=True
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            return False
        return True
    
    @staticmethod
    def logout():
        """Logout seguro"""
        keys_to_keep = ['password_correct']  # Chaves que devem ser mantidas após logout
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.session_state["password_correct"] = False
        st.rerun()

# Carregar configurações
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
TESSERACT_PATH = os.environ.get("TESSERACT_PATH") or st.secrets.get("TESSERACT_PATH", "")

# Configurar API Key do OpenAI
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Configurar Tesseract
if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Verificar autenticação
if not AuthManager.check_password():
    st.stop()

# Criar tabelas
db_manager.create_tables()

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Chat Assistant Pro</h1>
    <p>Converse inteligentemente com seus documentos usando IA avançada</p>
</div>
""", unsafe_allow_html=True)

# Verificar configurações essenciais
if not OPENAI_API_KEY:
    st.markdown(
        '<div class="warning-box">⚠️ OPENAI_API_KEY não configurada. Verifique suas variáveis de ambiente.</div>', 
        unsafe_allow_html=True
    )
    st.stop()

if not LANGCHAIN_AVAILABLE:
    st.markdown(
        '<div class="warning-box">⚠️ LangChain não disponível. Instale as dependências necessárias.</div>', 
        unsafe_allow_html=True
    )
    st.stop()

# Barra superior com informações do usuário
col1, col2 = st.columns([4, 1])
with col1:
    user = st.session_state.get("authenticated_user", "Usuário")
    st.markdown(f'<div class="success-box">✅ Bem-vindo, {user}! Sistema operacional.</div>', unsafe_allow_html=True)
with col2:
    if st.button("🚪 Sair", use_container_width=True, help="Fazer logout do sistema"):
        AuthManager.logout()

# Inicializar estado da sessão
if "sessoes" not in st.session_state:
    st.session_state["sessoes"] = {}

if "sessao_atual" not in st.session_state:
    nova_sessao = f"Sessao_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state["sessoes"][nova_sessao] = {
        "historico": [],
        "vectorstore": None,
        "nome": "Nova Conversa"
    }
    st.session_state["sessao_atual"] = nova_sessao

def criar_nova_sessao():
    """Criar nova sessão de chat"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    nova_sessao = f"Sessao_{timestamp}"
    st.session_state["sessoes"][nova_sessao] = {
        "historico": [],
        "vectorstore": None,
        "nome": f"Chat {datetime.now().strftime('%H:%M')}"
    }
    st.session_state["sessao_atual"] = nova_sessao
    st.rerun()

def criar_vectorstore(documentos: List[Document]) -> Optional[Any]:
    """Criar vectorstore com configurações otimizadas"""
    if not documentos or not LANGCHAIN_AVAILABLE:
        return None
    
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documentos)
        
        if not chunks:
            st.warning("⚠️ Nenhum conteúdo válido encontrado nos documentos")
            return None
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        logger.info(f"Vectorstore criado com {len(chunks)} chunks")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Erro ao criar vectorstore: {e}")
        st.error(f"❌ Erro ao criar índice de busca: {str(e)}")
        return None

# Sidebar melhorada
with st.sidebar:
    # Gerenciamento de sessões
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 💬 Sessões de Chat")
    
    sessoes_keys = list(st.session_state["sessoes"].keys())
    sessao_atual = st.session_state["sessao_atual"]

    if sessoes_keys:
        opcoes_sessoes = []
        for key in sessoes_keys:
            nome = st.session_state["sessoes"][key].get("nome", key)
            opcoes_sessoes.append((key, nome))
        
        # Selectbox personalizado
        indices = [opt[0] for opt in opcoes_sessoes]
        nomes = [opt[1] for opt in opcoes_sessoes]
        
        try:
            idx_atual = indices.index(sessao_atual)
        except ValueError:
            idx_atual = 0
        
        sessao_selecionada = st.selectbox(
            "🔄 Sessão Ativa:",
            options=indices,
            index=idx_atual,
            format_func=lambda x: f"📝 {st.session_state['sessoes'][x].get('nome', x)}"
        )
        
        if sessao_selecionada != sessao_atual:
            st.session_state["sessao_atual"] = sessao_selecionada
            st.rerun()

    # Botões de ação
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Nova", use_container_width=True, help="Criar nova sessão"):
            criar_nova_sessao()
    with col2:
        if st.button("🗑️ Limpar", use_container_width=True, help="Limpar sessão atual"):
            if sessao_atual in st.session_state["sessoes"]:
                st.session_state["sessoes"][sessao_atual]["historico"] = []
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Upload de arquivos
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload de Documentos")
    
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    st.markdown("**Arraste arquivos aqui ou clique para selecionar**")
    uploaded_files = st.file_uploader(
        "📁 Escolher arquivos",
        type=["pdf", "docx", "doc", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=f"uploader_{sessao_atual}",
        help="Formatos suportados: PDF, Word, CSV, Imagens (PNG, JPG)"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        st.markdown(f'<div class="info-box">📁 {len(uploaded_files)} arquivo(s) selecionado(s)</div>', unsafe_allow_html=True)
        
        if st.button("🔄 Processar Documentos", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            todos_docs = []
            total_files = len(uploaded_files)
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"📖 Processando: {file.name}")
                progress_bar.progress((i + 1) / total_files)
                
                try:
                    docs = FileProcessor.process_file(file)
                    todos_docs.extend(docs)
                    
                    # Salvar arquivo no banco (se disponível)
                    # Implementar salvamento aqui se necessário
                    
                except Exception as e:
                    st.error(f"❌ Erro ao processar {file.name}: {e}")

            if todos_docs:
                status_text.text("🔍 Criando índice de busca inteligente...")
                vectorstore = criar_vectorstore(todos_docs)
                
                if vectorstore:
                    st.session_state["sessoes"][sessao_atual]["vectorstore"] = vectorstore
                    progress_bar.progress(1.0)
                    status_text.text("")
                    st.markdown(f'<div class="success-box">🎉 {len(todos_docs)} documentos processados!</div>', unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.error("❌ Falha ao criar índice de busca")
            else:
                st.error("❌ Nenhum documento foi processado com sucesso")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Estatísticas da sessão
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Estatísticas")
    
    sessao_data = st.session_state["sessoes"][st.session_state["sessao_atual"]]
    historico = sessao_data["historico"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💬 Mensagens", len(historico))
    with col2:
        vectorstore_status = "✅ Ativo" if sessao_data.get("vectorstore") else "❌ Inativo"
        st.metric("🧠 IA", vectorstore_status)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Área principal do chat
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Título da sessão atual
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("## 💬 Área de Conversa")
with col2:
    if st.button("🔄 Recarregar", help="Recarregar conversa"):
        st.rerun()

# Exibir histórico de chat
if historico:
    for i, (pergunta, resposta) in enumerate(historico):
        with st.chat_message("user", avatar="👤"):
            st.write(pergunta)
        with st.chat_message("assistant", avatar="🤖"):
            st.write(resposta)
else:
    # Mensagem de boas-vindas
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 16px; margin: 2rem 0;">
        <h3>👋 Olá! Como posso ajudar hoje?</h3>
        <p>Faça upload de documentos e comece a conversar comigo sobre eles!</p>
        <p><strong>Dicas:</strong></p>
        <ul style="text-align: left; display: inline-block; margin-top: 1rem;">
            <li>📄 Envie PDFs, Word, CSV ou imagens</li>
            <li>🔍 Faça perguntas específicas sobre o conteúdo</li>
            <li>💡 Peça resumos ou análises detalhadas</li>
            <li>🌟 Use linguagem natural para melhores resultados</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Input de pergunta
pergunta = st.chat_input("💭 Digite sua pergunta aqui...", key="chat_input")

if pergunta:
    # Adicionar pergunta ao chat
    with st.chat_message("user", avatar="👤"):
        st.write(pergunta)

    # Processar resposta
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🧠 Processando sua pergunta..."):
            try:
                vectorstore = sessao_data.get("vectorstore")
                
                if vectorstore and LANGCHAIN_AVAILABLE:
                    # Chat com contexto dos documentos
                    retriever = vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4}  # Aumentar número de documentos relevantes
                    )
                    
                    llm = ChatOpenAI(
                        temperature=0.3, 
                        model_name="gpt-4o-mini",  # Modelo mais eficiente
                        max_tokens=1000
                    )
                    
                    memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        output_key="answer"
                    )
                    
                    # Adicionar histórico à memória
                    for hist_pergunta, hist_resposta in historico[-5:]:  # Últimas 5 interações
                        memory.chat_memory.add_user_message(hist_pergunta)
                        memory.chat_memory.add_ai_message(hist_resposta)
                    
                    chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=retriever,
                        memory=memory,
                        verbose=False,
                        return_source_documents=True
                    )
                    
                    resultado = chain({"question": pergunta})
                    resposta = resultado["answer"]
                    
                    # Mostrar fontes se disponíveis
                    if resultado.get("source_documents"):
                        with st.expander("📚 Fontes consultadas", expanded=False):
                            for i, doc in enumerate(resultado["source_documents"][:3]):
                                st.write(f"**Fonte {i+1}:** {doc.metadata.get('source', 'Desconhecida')}")
                                st.write(f"```{doc.page_content[:200]}...```")
                    
                else:
                    # Chat sem contexto
                    if LANGCHAIN_AVAILABLE:
                        llm = ChatOpenAI(
                            temperature=0.7, 
                            model_name="gpt-4o-mini",
                            max_tokens=800
                        )
                        
                        # Criar contexto com histórico recente
                        contexto = ""
                        if historico:
                            contexto = "Histórico recente da conversa:\n"
                            for hist_p, hist_r in historico[-3:]:
                                contexto += f"Usuário: {hist_p}\nAssistente: {hist_r}\n\n"
                        
                        prompt_completo = f"{contexto}Pergunta atual: {pergunta}"
                        resposta_obj = llm.invoke(prompt_completo)
                        resposta = resposta_obj.content if hasattr(resposta_obj, 'content') else str(resposta_obj)
                    else:
                        resposta = "❌ Sistema indisponível. Verifique as configurações."

                st.write(resposta)
                historico.append((pergunta, resposta))
                
            except Exception as e:
                error_msg = f"❌ Erro ao processar: {str(e)}"
                logger.error(f"Erro no chat: {e}")
                st.error(error_msg)
                historico.append((pergunta, error_msg))

st.markdown('</div>', unsafe_allow_html=True)

# Painel de controle inferior
st.markdown("---")
st.markdown("### 🎛️ Painel de Controle")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="📈 Sessões Ativas", 
        value=len(st.session_state["sessoes"]),
        help="Número total de sessões de chat"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="💬 Mensagens", 
        value=len(historico),
        help="Mensagens na sessão atual"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    vectorstore_status = "✅ Ativo" if sessao_data.get("vectorstore") else "❌ Inativo"
    st.metric(
        label="🔍 Busca IA", 
        value=vectorstore_status,
        help="Status do sistema de busca inteligente"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    api_status = "✅ OK" if OPENAI_API_KEY else "❌ Erro"
    st.metric(
        label="🔑 API OpenAI", 
        value=api_status,
        help="Status da conexão com OpenAI"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Funcionalidades avançadas
with st.expander("⚙️ Configurações Avançadas", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Parâmetros do Modelo**")
        temperature = st.slider(
            "Criatividade", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3, 
            step=0.1,
            help="Controla a criatividade das respostas"
        )
        
        max_tokens = st.slider(
            "Tamanho da Resposta", 
            min_value=100, 
            max_value=2000, 
            value=1000, 
            step=100,
            help="Limite máximo de tokens por resposta"
        )
    
    with col2:
        st.markdown("**🔍 Configurações de Busca**")
        search_k = st.slider(
            "Documentos Relevantes", 
            min_value=1, 
            max_value=10, 
            value=4,
            help="Número de documentos a consultar"
        )
        
        chunk_size = st.slider(
            "Tamanho do Chunk", 
            min_value=500, 
            max_value=2000, 
            value=1000, 
            step=250,
            help="Tamanho dos fragmentos de texto"
        )

# Exportar conversa
if historico:
    st.markdown("---")
    st.markdown("### 📥 Exportar Conversa")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Exportar como Texto", use_container_width=True):
            texto_conversa = f"Conversa - {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
            for i, (p, r) in enumerate(historico, 1):
                texto_conversa += f"Pergunta {i}: {p}\n"
                texto_conversa += f"Resposta {i}: {r}\n\n"
            
            st.download_button(
                label="💾 Baixar Conversa",
                data=texto_conversa,
                file_name=f"conversa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("📊 Exportar como JSON", use_container_width=True):
            dados_conversa = {
                "sessao": st.session_state["sessao_atual"],
                "timestamp": datetime.now().isoformat(),
                "total_mensagens": len(historico),
                "conversa": [
                    {"pergunta": p, "resposta": r, "indice": i} 
                    for i, (p, r) in enumerate(historico, 1)
                ]
            }
            
            st.download_button(
                label="💾 Baixar JSON",
                data=json.dumps(dados_conversa, indent=2, ensure_ascii=False),
                file_name=f"conversa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("🧹 Limpar Histórico", use_container_width=True):
            st.session_state["sessoes"][st.session_state["sessao_atual"]]["historico"] = []
            st.success("✅ Histórico limpo!")
            st.rerun()

# Footer informativo
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 16px; margin-top: 2rem;">
    <p><strong>🤖 AI Chat Assistant Pro v2.0</strong></p>
    <p>Sistema inteligente de conversação com documentos • Powered by OpenAI GPT-4</p>
    <p><small>💡 Dica: Para melhores resultados, seja específico em suas perguntas e use documentos bem estruturados.</small></p>
</div>
""", unsafe_allow_html=True)

# Debug info (apenas para desenvolvimento)
if st.secrets.get("DEBUG_MODE", False):
    with st.expander("🔧 Debug Info", expanded=False):
        st.json({
            "sessao_atual": st.session_state["sessao_atual"],
            "total_sessoes": len(st.session_state["sessoes"]),
            "vectorstore_ativo": bool(sessao_data.get("vectorstore")),
            "langchain_disponivel": LANGCHAIN_AVAILABLE,
            "openai_key_configurada": bool(OPENAI_API_KEY),
            "total_mensagens": len(historico)
        })
