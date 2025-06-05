import streamlit as st
import os
import tempfile
from PIL import Image
import pytesseract
from datetime import datetime
import psycopg2
from urllib.parse import urlparse
import json
import logging
from typing import Optional, List, Tuple, Any
from contextlib import contextmanager

# Imports do LangChain
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

# Configuração do Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="🤖 AI Chat Assistant Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado (estilos completos para a UI)
st.markdown("""
<style>
/* Reset e base */
* { margin:0; padding:0; box-sizing:border-box; }
/* Variáveis */
:root {
  --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  --success-gradient: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
  --warning-gradient: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
  --glass-bg: rgba(255, 255, 255, 0.1);
  --glass-border: rgba(255, 255, 255, 0.2);
  --shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
  --border-radius: 16px;
}
/* Header principal */
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
  top: 0; left: 0; right: 0; bottom: 0;
  background: url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M20 20c0 11.046-8.954 20-20 20s-20-8.954-20-20 8.954-20 20-20 20 8.954 20 20zm-10 0c0 5.523-4.477 10-10 10s-10-4.477-10-10 4.477-10 10-10 10 4.477 10 10z'/%3E%3C/g%3E%3C/svg%3E");
}
.main-header h1 { font-size: 2.5rem; font-weight: 700; position: relative; z-index: 1; }
.main-header p { font-size: 1.1rem; opacity: 0.9; position: relative; z-index: 1; }
/* Containers */
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
/* Botões */
.stButton > button {
  background: var(--primary-gradient) !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.8rem 1.5rem !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
}
/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.1); border-radius: 10px; }
::-webkit-scrollbar-thumb { background: var(--primary-gradient); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: linear-gradient(135deg, #5a6fd8 0%, #6b5b95 100%); }
</style>
""", unsafe_allow_html=True)

# Configurações iniciais: chaves de API e caminhos
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
TESSERACT_PATH  = os.environ.get("TESSERACT_PATH")  or st.secrets.get("TESSERACT_PATH", "")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Configurar proxies via variáveis de ambiente, se necessário
http_proxy  = os.environ.get("HTTP_PROXY")  or os.environ.get("http_proxy")
https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
PROXIES = {}
if http_proxy or https_proxy:
    PROXIES = {"http": http_proxy, "https": https_proxy}
    logger.info(f"Usando proxy: {PROXIES}")

# ----- DatabaseManager -----
class DatabaseManager:
    """Gerenciador de conexões com o banco de dados."""
    def __init__(self):
        self.connection_params = self._parse_database_url()

    def _parse_database_url(self) -> Optional[dict]:
        default_db_url = (
            "postgres://postgres:"
            "kNL6exzv6Y3HYomX4Etgpb2fqtatWIuzKh5OYozkM9NkayOywHe1i1jfyvijgS3G"
            "@185.173.110.61:9898/postgres"
        )
        db_url = os.environ.get("DATABASE_URL") or st.secrets.get("DATABASE_URL", default_db_url)
        if not db_url:
            logger.warning("DATABASE_URL não encontrada")
            return None
        try:
            parsed = urlparse(db_url)
            try:
                port = parsed.port
            except ValueError:
                port = 9898
            params = {
                "host":     parsed.hostname,
                "port":     int(port or 9898),
                "database": parsed.path.lstrip("/"),
                "user":     parsed.username,
                "password": parsed.password
            }
            if not all([params["host"], params["database"], params["user"]]):
                raise ValueError("Parâmetros de conexão incompletos")
            logger.info(f"Conexão configurada para: {params['host']}:{params['port']}/{params['database']}")
            return params
        except Exception as e:
            logger.error(f"Erro ao parsear DATABASE_URL: {e}")
            st.error(f"❌ Erro na configuração do banco: {e}")
            return None

    @contextmanager
    def get_connection(self):
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
        finally:
            if conn:
                conn.close()

    def create_tables(self) -> bool:
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
            );
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
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS users (
              id SERIAL PRIMARY KEY,
              username VARCHAR(50) UNIQUE NOT NULL,
              password_hash VARCHAR(255) NOT NULL,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              last_login TIMESTAMP,
              is_active BOOLEAN DEFAULT TRUE
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_file_storage_hash ON file_storage(file_hash);",
            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_active ON chat_sessions(is_active, updated_at);",
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username) WHERE is_active = TRUE;"
        ]
        with self.get_connection() as conn:
            if not conn:
                return False
            try:
                with conn.cursor() as cur:
                    for cmd in commands:
                        cur.execute(cmd)
                conn.commit()
                logger.info("Tabelas criadas/atualizadas com sucesso")
                return True
            except Exception as e:
                logger.error(f"Erro ao criar tabelas: {e}")
                st.error(f"❌ Erro ao criar tabelas: {e}")
                return False

db_manager = DatabaseManager()

# ----- FileProcessor -----
class FileProcessor:
    """Processador de arquivos para extração de conteúdo."""
    @staticmethod
    def format_file_size(size: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    @staticmethod
    def process_file(file) -> List[Document]:
        if not LANGCHAIN_AVAILABLE:
            st.error("❌ LangChain não disponível")
            return []
        ext = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name
        docs: List[Document] = []
        try:
            if ext == "pdf":
                loader = PyMuPDFLoader(tmp_path)
                docs = loader.load()
            elif ext in ("docx", "doc"):
                loader = Docx2txtLoader(tmp_path)
                docs = loader.load()
            elif ext == "csv":
                loader = CSVLoader(tmp_path, encoding="utf-8")
                docs = loader.load()
            elif ext in ("png", "jpg", "jpeg"):
                img = Image.open(tmp_path)
                text = pytesseract.image_to_string(img, lang="por+eng")
                if text.strip():
                    docs = [Document(page_content=text, metadata={"source": file.name, "type": "image_ocr"})]
                else:
                    st.warning(f"⚠️ Nenhum texto encontrado em {file.name}")
            else:
                st.error(f"❌ Tipo de arquivo não suportado: {ext}")
        except Exception as e:
            logger.error(f"Erro ao processar {file.name}: {e}")
            st.error(f"❌ Erro ao processar {file.name}: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return docs

# ----- AuthManager -----
class AuthManager:
    """Gerenciador de autenticação simples."""
    @staticmethod
    def check_password() -> bool:
        def on_submit():
            u = st.session_state.get("username", "").strip()
            p = st.session_state.get("password", "")
            if u == "Hisoka" and p == "Hisoka123#":
                st.session_state["password_correct"] = True
                st.session_state["authenticated_user"] = u
                for k in ("username", "password"):
                    st.session_state.pop(k, None)
            else:
                st.session_state["password_correct"] = False

        if "password_correct" not in st.session_state:
            st.session_state["password_correct"] = False

        if not st.session_state["password_correct"]:
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.markdown('<div class="login-container">', unsafe_allow_html=True)
                st.markdown("# 🔐 AI Chat Login")
                with st.form("login"):
                    st.text_input("👤 Usuário", key="username", placeholder="Digite seu usuário")
                    st.text_input("🔒 Senha", key="password", type="password", placeholder="Digite sua senha")
                    st.form_submit_button("🚀 Entrar", on_click=on_submit)
                if st.session_state["password_correct"] is False:
                    st.markdown('<div class="warning-box">😕 Credenciais incorretas.</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            return False
        return True

    @staticmethod
    def logout():
        keep = ["password_correct"]
        for k in list(st.session_state.keys()):
            if k not in keep:
                del st.session_state[k]
        st.session_state["password_correct"] = False
        st.rerun()

# Verificação de autenticação e criação de tabelas
if not AuthManager.check_password():
    st.stop()

db_manager.create_tables()

# ----- Header -----
st.markdown("""
<div class="main-header">
  <h1>🤖 AI Chat Assistant Pro</h1>
  <p>Converse inteligentemente com seus documentos usando IA avançada</p>
</div>
""", unsafe_allow_html=True)

if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY não configurada. Verifique suas variáveis de ambiente.")
    st.stop()

if not LANGCHAIN_AVAILABLE:
    st.error("⚠️ LangChain não disponível. Instale as dependências necessárias.")
    st.stop()

# ----- Top Bar -----
col1, col2 = st.columns([4, 1])
with col1:
    user = st.session_state.get("authenticated_user", "Usuário")
    st.success(f"✅ Bem-vindo, {user}!")
with col2:
    if st.button("🚪 Sair"):
        AuthManager.logout()

# Gerenciamento de sessões
if "sessoes" not in st.session_state:
    st.session_state["sessoes"] = {}
if "sessao_atual" not in st.session_state:
    sessao_id = datetime.now().strftime("Sessao_%Y%m%d_%H%M%S")
    st.session_state["sessoes"][sessao_id] = {"historico": [], "vectorstore": None, "nome": "Nova Conversa"}
    st.session_state["sessao_atual"] = sessao_id

def criar_nova_sessao():
    novo_id = datetime.now().strftime("Sessao_%Y%m%d_%H%M%S")
    st.session_state["sessoes"][novo_id] = {"historico": [], "vectorstore": None, "nome": f"Chat {datetime.now().strftime('%H:%M')}"}
    st.session_state["sessao_atual"] = novo_id
    st.rerun()

# ----- Sidebar -----
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 💬 Sessões de Chat")
    keys = list(st.session_state["sessoes"].keys())
    sessao_atual = st.session_state["sessao_atual"]
    if keys:
        sessao_selecionada = st.selectbox(
            "🔄 Sessão Ativa:",
            options=keys,
            index=keys.index(sessao_atual),
            format_func=lambda x: st.session_state["sessoes"][x]["nome"]
        )
        if sessao_selecionada != sessao_atual:
            st.session_state["sessao_atual"] = sessao_selecionada
            st.rerun()
    colA, colB = st.columns(2)
    with colA:
        if st.button("➕ Nova"):
            criar_nova_sessao()
    with colB:
        if st.button("🗑️ Limpar"):
            st.session_state["sessoes"][sessao_atual]["historico"] = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Upload de documentos
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload de Documentos")
    uploaded_files = st.file_uploader(
        "Escolha arquivos",
        type=["pdf", "docx", "doc", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=f"uploader_{sessao_atual}"
    )
    if uploaded_files:
        st.info(f"{len(uploaded_files)} arquivo(s) selecionado(s)")
        if st.button("🔄 Processar Documentos"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            todos_docs: List[Document] = []
            total = len(uploaded_files)
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processando: {file.name}")
                progress_bar.progress((i + 1) / total)
                docs = FileProcessor.process_file(file)
                todos_docs.extend(docs)
            if todos_docs:
                status_text.text("Criando índice de busca...")
                vectorstore = criar_vectorstore(todos_docs)
                if vectorstore:
                    st.session_state["sessoes"][sessao_atual]["vectorstore"] = vectorstore
                    progress_bar.progress(1.0)
                    status_text.text("")
                    st.success(f"{len(todos_docs)} documentos processados com sucesso!")
                    st.balloons()
                else:
                    st.error("❌ Falha ao criar índice de busca")
            else:
                st.error("❌ Nenhum documento foi processado com sucesso")
    st.markdown('</div>', unsafe_allow_html=True)

    # Estatísticas da sessão
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Estatísticas")
    sessao_data = st.session_state["sessoes"][sessao_atual]
    historico = sessao_data["historico"]
    st.metric("💬 Mensagens", len(historico))
    st.metric("🔍 Busca IA", "✅ Ativo" if sessao_data.get("vectorstore") else "❌ Inativo")
    st.markdown('</div>', unsafe_allow_html=True)

# ----- Funções de IA e criação de modelo -----
def criar_vectorstore(documentos: List[Document]) -> Optional[Any]:
    if not documentos or not LANGCHAIN_AVAILABLE:
        return None
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documentos)
        if not chunks:
            st.warning("⚠️ Nenhum conteúdo válido encontrado.")
            return None

        # Se proxies estiverem configurados, passar via client_kwargs
        emb_kwargs = {}
        if PROXIES:
            emb_kwargs["client_kwargs"] = {"proxies": PROXIES}

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", **emb_kwargs)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        logger.info(f"Vectorstore criado com {len(chunks)} chunks")
        return vectorstore
    except Exception as e:
        logger.error(f"Erro ao criar vectorstore: {e}")
        st.error(f"❌ Erro ao criar índice de busca: {e}")
        return None

def create_llm() -> ChatOpenAI:
    """
    Instancia o ChatOpenAI utilizando o modelo gpt-4o-mini.
    Se proxies estiverem configurados, eles serão repassados via client_kwargs.
    """
    kwargs = {
        "temperature": 0.3,
        "model_name": "gpt-4o-mini",
        "max_tokens": 1000
    }
    if PROXIES:
        kwargs["client_kwargs"] = {"proxies": PROXIES}
    return ChatOpenAI(**kwargs)

# ----- Área de Chat Principal -----
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
sessao_atual = st.session_state["sessao_atual"]
sessao_data = st.session_state["sessoes"][sessao_atual]
historico: List[Tuple[str, str]] = sessao_data["historico"]

if historico:
    for pergunta_antiga, resposta_antiga in historico:
        with st.chat_message("user"):
            st.write(pergunta_antiga)
        with st.chat_message("assistant"):
            st.write(resposta_antiga)
else:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%); border-radius: 16px;">
      <h3>👋 Olá! Envie documentos e comece a conversar.</h3>
      <p>Faça upload de arquivos ou digite sua pergunta abaixo.</p>
    </div>
    """, unsafe_allow_html=True)

pergunta = st.chat_input("💭 Digite sua pergunta aqui...", key="chat_input")
if pergunta:
    with st.chat_message("user"):
        st.write(pergunta)
    with st.chat_message("assistant"):
        with st.spinner("🧠 Processando sua pergunta..."):
            try:
                vectorstore = sessao_data.get("vectorstore")
                if vectorstore and LANGCHAIN_AVAILABLE:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                    llm = create_llm()
                    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
                    # Adiciona histórico à memória (últimas 5 interações)
                    for hist_q, hist_a in historico[-5:]:
                        memory.chat_memory.add_user_message(hist_q)
                        memory.chat_memory.add_ai_message(hist_a)
                    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
                    result = chain({"question": pergunta})
                    resposta = result["answer"]
                    sources = result.get("source_documents", [])
                    if sources:
                        with st.expander("📚 Fontes consultadas", expanded=False):
                            for i, doc in enumerate(sources[:3], start=1):
                                st.write(f"**Fonte {i}:** {doc.metadata.get('source','Desconhecida')}")
                                st.write(doc.page_content[:200] + "...")
                else:
                    llm = create_llm()
                    # Cria um contexto simples com as últimas 3 interações
                    contexto = ""
                    for hist_q, hist_a in historico[-3:]:
                        contexto += f"Usuário: {hist_q}\nAssistente: {hist_a}\n"
                    prompt = f"{contexto}\nPergunta: {pergunta}"
                    resposta_obj = llm.invoke(prompt)
                    resposta = getattr(resposta_obj, "content", str(resposta_obj))
                st.write(resposta)
                historico.append((pergunta, resposta))
            except Exception as e:
                error_msg = f"❌ Erro ao processar: {e}"
                logger.error(error_msg)
                st.error(error_msg)
                historico.append((pergunta, error_msg))
st.markdown('</div>', unsafe_allow_html=True)

# ----- Exportar Conversa -----
if historico:
    st.markdown("---")
    st.markdown("### 📥 Exportar Conversa")
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    txt_conversa = f"Conversa - {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
    for idx, (q, a) in enumerate(historico, start=1):
        txt_conversa += f"Pergunta {idx}: {q}\nResposta {idx}: {a}\n\n"
    with col_exp1:
        st.download_button(
            label="💾 Exportar como TXT",
            data=txt_conversa,
            file_name=f"conversa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    with col_exp2:
        data_json = {
            "sessao": sessao_atual,
            "timestamp": datetime.now().isoformat(),
            "mensagens": len(historico),
            "conversa": [{"pergunta": q, "resposta": a} for q, a in historico]
        }
        st.download_button(
            label="💾 Exportar como JSON",
            data=json.dumps(data_json, indent=2, ensure_ascii=False),
            file_name=f"conversa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    with col_exp3:
        if st.button("🧹 Limpar Histórico"):
            st.session_state["sessoes"][sessao_atual]["historico"] = []
            st.success("✅ Histórico limpo!")
            st.rerun()

# ----- Footer -----
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%); border-radius: 16px; margin-top: 2rem;">
  <p><strong>🤖 AI Chat Assistant Pro v2.0</strong></p>
  <p>Sistema de conversação com documentos • Powered by OpenAI GPT-4</p>
</div>
""", unsafe_allow_html=True)

# ----- Debug (opcional) -----
if st.secrets.get("DEBUG_MODE", False):
    with st.expander("🔧 Debug Info", expanded=False):
        st.json({
            "sessao_atual": sessao_atual,
            "total_sessoes": len(st.session_state["sessoes"]),
            "vectorstore_ativo": bool(sessao_data.get("vectorstore")),
            "langchain_disponivel": LANGCHAIN_AVAILABLE,
            "openai_key_configurada": bool(OPENAI_API_KEY),
            "total_mensagens": len(historico)
        })
