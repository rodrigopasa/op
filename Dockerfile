# Usando uma imagem base Python
FROM python:3.11-slim

# Diretório de trabalho
WORKDIR /app

# Copie o arquivo requirements.txt primeiro
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie todo o restante do código
COPY . .

# Exponha a porta padrão do Streamlit
EXPOSE 8501

# Comando para rodar o seu app
CMD ["streamlit", "run", "chat.py", "--server.port=8501", "--server.address=0.0.0.0"]
