FROM adefe/strawberry_env:v2

WORKDIR /home

COPY . /home

CMD ["uvicorn", "--app-dir", "src", "--host", "0.0.0.0", "--port", "20003", "server:app"]
