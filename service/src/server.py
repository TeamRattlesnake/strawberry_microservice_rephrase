import os
import copy
import logging
import traceback
import sys
from multiprocessing import Process
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from models import ResponseModel, AddGroupModel, GenerateModel
from logic import NeuralNetwork


logging.basicConfig(format="%(asctime)s %(message)s", handlers=[logging.FileHandler(
    f"/home/logs/rephrase_log.txt", mode="w", encoding="UTF-8")], datefmt="%I:%M:%S %p", level=logging.INFO)


app = FastAPI()


NN = None
process_pool = {}

DESCRIPTION = """
Микросервис для Strawberry
"""


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Strawberry Microservice",
        version="0.1.0",
        description=DESCRIPTION,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.on_event("startup")
def startup():
    '''Функция, запускающаяся при старте сервера'''
    logging.info("Server started")
    global NN
    logging.info("Creating primary NeuralNetwork")
    NN = NeuralNetwork()
    logging.info("Primary NeuralNetwork is ready")


@app.post("/add_group", response_model=ResponseModel)
async def add_group(data: AddGroupModel):
    group_id = data.group_id
    texts = data.texts
    logging.info(f"Adding group {group_id}")
    if len(texts) == 0:
        raise ValueError("Empty texts (who cares)")
    return ResponseModel(result="OK")


@app.post("/generate", response_model=ResponseModel)
async def generate(data: GenerateModel):
    group_id = data.group_id
    hint = data.hint
    logging.info(f"Generating content for group {group_id}")
    try:
        result = NN.generate(hint)
        return ResponseModel(result=result)
    except Exception as e:
        logging.error(e)
        logging.info(traceback.format_exc())
        return ResponseModel(result="ERROR")


@app.get("/check_status", response_model=ResponseModel)
async def check_status(group_id: int):
    logging.info(f"Cheking status for group {group_id}")
    return ResponseModel(result="OK")  # Всегда готова
