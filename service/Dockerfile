FROM debian:bullseye-slim

WORKDIR /home

COPY . /home

RUN apt update && \
    apt install -y python3 python3-pip && \
    pip3 install -r requirements.txt



CMD ["uvicorn", "--app-dir", "src", "--host", "0.0.0.0", "--port", "20003", "server:app"]
