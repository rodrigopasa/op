FROM python:3.11-slim

WORKDIR /app

# Copie o arquivo requirements.txt primeiro
COPY requirements.txt .

# Depois copie o restante dos arquivos
COPY . .

# Instale dependências
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "chat.py", "--server.port=8501", "--server.address=0.0.0.0"]
