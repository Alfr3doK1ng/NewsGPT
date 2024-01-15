FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/streamlit/streamlit-example.git .

COPY ./requirements.txt .
COPY ./app.py .

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]

CMD ["app.py"] 