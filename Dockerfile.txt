FROM python:3.11-slim

WORKDIR /app

# Instala o Tesseract OCR
RUN apt-get update && apt-get install -y tesseract-ocr

# Copie seu código
COPY . .

# Instale dependências
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "seu_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
