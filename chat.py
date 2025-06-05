import streamlit as st
import os
import tempfile
from PIL import Image
import pytesseract
from datetime import datetime
import psycopg2
from psycopg2 import sql
import json
from urllib.parse import urlparse
import logging
from typing import Optional, List, Any, Tuple
from contextlib import contextmanager

# LangChain imports
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

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="🤖 AI Chat Assistant Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
/* Reset e base */
* { margin:0; padding:0; box-sizing:border-box; }
/* Variáveis */
:root {
  --primary-gradient: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  --glass-bg: rgba(255,255,255,0.1);
  --glass-border: rgba(255,255,255,0.2);
  --shadow: 0 8px 32px rgba(31,38,135,0.37);
  --border-radius: 16px;
}
/* Header */
.main-header {
  background: var(--primary-gradient);
  padding:2rem; border-radius:var(--border-radius);
  color:#fff; text-align:center; margin-bottom:2rem;
  box-shadow:var(--shadow); backdrop-filter:blur(10px);
  position:relative; overflow:hidden;
}
.main-header::before {
  content:''; position:absolute; top:0; left:0; right:0; bottom:0;
  background:url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M20 20c0 11.046-8.954 20-20 20s-20-8.954-20-20 8.954-20 20-20 20 8.954 20 20zm-10 0c0 5.523-4.477 10-10 10s-10-4.477-10-10 4.477-10 10-10 10 4.477 10 10z'/%3E%3C/g%3E%3C/svg%3E");
}
.main-header h1 { font-size:2.5rem; font-weight:700; z-index:1; }
.main-header p { font-size:1.1rem; opacity:0.9; z-index:1; }
/* Containers */
.chat-container, .sidebar-section {
  background: var(--glass-bg); backdrop-filter:blur(10px);
  border:1px solid var(--glass-border); border-radius:var(--border-radius);
  padding:1.5rem; margin:1rem 0; box-shadow:var(--shadow);
}
.chat-container { min-height:400px; background:linear-gradient(135deg,rgba(255,255,255,0.1) 0%,rgba(255,255,255,0.05) 100%); }
/* Buttons */
.stButton > button {
  background:var(--primary-gradient)!important;
  color:#fff!important; border:none!important;
  border-radius:12px!important; padding:0.8rem 1.5rem!important;
  font-weight:600!important; font-size:0.95rem!important;
  box-shadow:0 4px 15px rgba(102,126,234,0.4)!important;
}
.stButton > button:hover {
  transform:translateY(-2px)!important;
  box-shadow:0 8px 25px rgba(102,126,234,0.6)!important;
}
/* Scrollbar */
::-webkit-scrollbar { width:8px; }
::-webkit-scrollbar-track { background:rgba(255,255,255,0.1); border-radius:10px; }
::-webkit-scrollbar-thumb { background:var(--primary-gradient); border-radius:10px; }
::-webkit-scrollbar-thumb:hover { background:linear-gradient(135deg,#5a6fd8 0%,#6b5b95 100%); }
</style>
""", unsafe_allow_html=True)

# Configurações iniciais
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
TESSERACT_PATH  = os.environ.get("TESSERACT_PATH")  or st.secrets.get("TESSERACT_PATH", "")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Configurar proxies via env
http_proxy  = os.environ.get("HTTP_PROXY")  or os.environ.get("http_proxy")
https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
PROXIES = {}
if http_proxy or https_proxy:
    PROXIES = {"http": http_proxy, "https": https_proxy}
    logger.info(f"Usando proxy: {PROXIES}")

# ===== DatabaseManager =====
class DatabaseManager:
    """Gerenciador de conexões com banco de dados."""
    def __init__(self):
        self.connection_params = self._parse_database_url()

    def _parse_database_url(self) -> Optional[dict]:
        default = (
            "postgres://postgres:"
            "kNL6exzv6Y3HYomX4Etgpb2fqtatWIuzKh5OYozkM9NkayOywHe1i1jfyvijgS3G"
            "@185.173.110.61:9898/postgres"
        )
        db_url = os.environ.get("DATABASE_URL") or st.secrets.get("DATABASE_URL", default)
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
            logger.info(f"Conexão: {params['host']}:{params['port']}/{params['database']}")
            return params
        except Exception as e:
            logger.error(f"Erro parse DATABASE_URL: {e}")
            st.error(f"❌ Configuração do banco: {e}")
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
            logger.error(f"Erro conexão PostgreSQL: {e}")
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
                logger.info("Tabelas criadas/atualizadas")
                return True
            except Exception as e:
                logger.error(f"Erro criar tabelas: {e}")
                st.error(f"❌ Erro criar tabelas: {e}")
                return False

db_manager = DatabaseManager()

# ===== FileProcessor =====
class FileProcessor:
    """Processador de arquivos."""
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
                    docs = [Document(page_content=text,
                                     metadata={"source": file.name, "type": "image_ocr"})]
                else:
                    st.warning(f"⚠️ Nenhum texto em {file.name}")
            else:
                st.error(f"❌ Tipo não suportado: {ext}")
        except Exception as e:
            logger.error(f"Erro processar {file.name}: {e}")
            st.error(f"❌ Erro processar {file.name}: {e}")
        finally:
            try: os.remove(tmp_path)
            except: pass
        return docs

# ===== AuthManager =====
class AuthManager:
    """Gerenciador de autenticação."""
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
                    st.text_input("👤 Usuário", key="username")
                    st.text_input("🔒 Senha", key="password", type="password")
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

# Autenticação e DB
if not AuthManager.check_password():
    st.stop()
db_manager.create_tables()

# Header
st.markdown("""
<div class="main-header">
  <h1>🤖 AI Chat Assistant Pro</h1>
  <p>Converse inteligentemente com seus documentos</p>
</div>
""", unsafe_allow_html=True)

# Verificações
if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY não configurada."); st.stop()
if not LANGCHAIN_AVAILABLE:
    st.error("⚠️ LangChain não disponível."); st.stop()

# Top bar
col1, col2 = st.columns([4, 1])
with col1:
    user = st.session_state.get("authenticated_user", "Usuário")
    st.success(f"✅ Bem-vindo, {user}!")
with col2:
    if st.button("🚪 Sair"):
        AuthManager.logout()

# Sessões
if "sessoes" not in st.session_state:
    st.session_state["sessoes"] = {}
if "sessao_atual" not in st.session_state:
    sid = datetime.now().strftime("Sessao_%Y%m%d_%H%M%S")
    st.session_state["sessoes"][sid] = {"historico": [], "vectorstore": None, "nome": "Nova Conversa"}
    st.session_state["sessao_atual"] = sid

def criar_nova_sessao():
    ts = datetime.now().strftime("Sessao_%Y%m%d_%H%M%S")
    st.session_state["sessoes"][ts] = {"historico": [], "vectorstore": None, "nome": f"Chat {datetime.now().strftime('%H:%M')}"}
    st.session_state["sessao_atual"] = ts
    st.rerun()

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 💬 Sessões de Chat")
    keys = list(st.session_state["sessoes"].keys())
    current = st.session_state["sessao_atual"]
    if keys:
        sel = st.selectbox("🔄 Sessão Ativa:", keys, index=keys.index(current),
                           format_func=lambda x: st.session_state["sessoes"][x]["nome"])
        if sel != current:
            st.session_state["sessao_atual"] = sel
            st.rerun()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ Nova"):
            criar_nova_sessao()
    with c2:
        if st.button("🗑️ Limpar"):
            st.session_state["sessoes"][current]["historico"] = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload de Documentos")
    uploaded = st.file_uploader("Escolher arquivos", type=["pdf","docx","doc","csv","png","jpg","jpeg"],
                                accept_multiple_files=True, key=f"up_{current}")
    if uploaded:
        st.info(f"{len(uploaded)} arquivo(s) selecionado(s)")
        if st.button("🔄 Processar Documentos"):
            bar = st.progress(0); status = st.empty()
            all_docs: List[Document] = []
            for i, f in enumerate(uploaded):
                status.text(f"Processando {f.name}")
                bar.progress((i+1)/len(uploaded))
                docs = FileProcessor.process_file(f)
                all_docs.extend(docs)
            if all_docs:
                status.text("Criando índice...")
                # criar_vectorstore abaixo
                bar.progress(1.0)
                st.session_state["sessoes"][current]["vectorstore"] = criar_vectorstore(all_docs)
                st.success(f"{len(all_docs)} docs processados!")
                st.balloons()
            else:
                st.error("❌ Nenhum documento processado")
    st.markdown('</div>', unsafe_allow_html=True)

    # Estatísticas
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Estatísticas")
    sess = st.session_state["sessoes"][current]
    hist = sess["historico"]
    st.metric("💬 Mensagens", len(hist))
    st.metric("🔍 Busca IA", "✅" if sess["vectorstore"] else "❌")
    st.markdown('</div>', unsafe_allow_html=True)

# Funções de IA
def criar_vectorstore(docs: List[Document]) -> Optional[Any]:
    if not docs or not LANGCHAIN_AVAILABLE:
        return None
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        if not chunks:
            st.warning("⚠️ Sem conteúdo válido")
            return None

        # Embeddings sem proxies explícitos
        emb_kwargs = {}
        if PROXIES:
            emb_kwargs["client_kwargs"] = {"proxies": PROXIES}

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", **emb_kwargs)
        vs = FAISS.from_documents(chunks, embeddings)
        logger.info(f"Vectorstore criado com {len(chunks)} chunks")
        return vs
    except Exception as e:
        logger.error(f"Erro criar vectorstore: {e}")
        st.error(f"❌ Erro ao criar índice: {e}")
        return None

def create_llm() -> ChatOpenAI:
    """Instancia ChatOpenAI usando gpt-4o-mini e proxies se configurados."""
    kwargs = {"temperature": 0.3, "model_name": "gpt-4o-mini", "max_tokens": 1000}
    if PROXIES:
        kwargs["client_kwargs"] = {"proxies": PROXIES}
    return ChatOpenAI(**kwargs)

# Chat area
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
sess = st.session_state["sessoes"][st.session_state["sessao_atual"]]
historico: List[Tuple[str,str]] = sess["historico"]

# Exibir histórico
if historico:
    for q, a in historico:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
else:
    st.markdown("""
    <div style="text-align:center; padding:2rem; background:rgba(118,75,162,0.1); border-radius:16px;">
      <h3>👋 Olá! Envie documentos e converse com eles.</h3>
    </div>
    """, unsafe_allow_html=True)

# Pergunta
pergunta = st.chat_input("Digite sua pergunta...")
if pergunta:
    with st.chat_message("user"):
        st.write(pergunta)
    with st.chat_message("assistant"):
        with st.spinner("🧠 Processando..."):
            try:
                vectorstore = sess["vectorstore"]
                if vectorstore and LANGCHAIN_AVAILABLE:
                    retr = vectorstore.as_retriever(search_kwargs={"k": 4})
                    llm = create_llm()
                    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
                    # popular memória
                    for pq, pa in historico[-5:]:
                        memory.chat_memory.add_user_message(pq)
                        memory.chat_memory.add_ai_message(pa)
                    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retr, memory=memory)
                    result = chain({"question": pergunta})
                    resposta = result["answer"]
                    sources = result.get("source_documents", [])
                    if sources:
                        with st.expander("📚 Fontes"):
                            for i, doc in enumerate(sources[:3], 1):
                                st.write(f"**Fonte {i}:** {doc.metadata.get('source','?')}")
                                st.write(doc.page_content[:200] + "...")
                else:
                    llm = create_llm()
                    # contexto simples
                    contexto = ""
                    for pq, pa in historico[-3:]:
                        contexto += f"Usuário: {pq}\nAssistente: {pa}\n"
                    prompt = f"{contexto}\nPergunta: {pergunta}"
                    resp_obj = llm.invoke(prompt)
                    resposta = getattr(resp_obj, "content", str(resp_obj))
                st.write(resposta)
                historico.append((pergunta, resposta))
            except Exception as e:
                msg = f"❌ Erro ao processar: {e}"
                logger.error(msg)
                st.error(msg)
                historico.append((pergunta, msg))
st.markdown('</div>', unsafe_allow_html=True)

# Exportar conversa
if historico:
    st.markdown("---")
    st.markdown("### 📥 Exportar Conversa")
    c1, c2, c3 = st.columns(3)
    texto = "\n".join(f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(historico))
    with c1:
        st.download_button("💾 TXT", data=texto, file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with c2:
        data = {
            "sessao": st.session_state["sessao_atual"],
            "timestamp": datetime.now().isoformat(),
            "conversa": [{"pergunta":q, "resposta":a} for q, a in historico]
        }
        st.download_button("💾 JSON", data=json.dumps(data, ensure_ascii=False, indent=2),
                           file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with c3:
        if st.button("🧹 Limpar Histórico"):
            st.session_state["sessoes"][st.session_state["sessao_atual"]]["historico"] = []
            st.success("Histórico limpo!")
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:1rem; background:rgba(118,75,162,0.1); border-radius:16px;">
  <p><strong>🤖 AI Chat Assistant Pro v2.0</strong></p>
  <p>Powered by OpenAI GPT-4</p>
</div>
""", unsafe_allow_html=True)

# Debug
if st.secrets.get("DEBUG_MODE", False):
    with st.expander("🔧 Debug"):
        st.json({
            "sessao_atual": st.session_state["sessao_atual"],
            "total_sessoes": len(st.session_state["sessoes"]),
            "vectorstore": bool(sess["vectorstore"]),
            "langchain": LANGCHAIN_AVAILABLE,
            "openai_key": bool(OPENAI_API_KEY),
            "mensagens": len(historico)
        })
