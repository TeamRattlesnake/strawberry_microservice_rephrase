import os
import re

import random
from tqdm import tqdm
import logging

import torch

from transformers import pipeline

logging.basicConfig(format="%(asctime)s %(message)s", handlers=[logging.FileHandler(
    f"/home/logs/rephrase_log_model.txt", mode="w", encoding="UTF-8")], datefmt="%I:%M:%S %p", level=logging.INFO)


class NeuralNetwork:
    def __init__(self, group_id=0):
       self.pipe = pipeline(model="cointegrated/rut5-base-paraphraser")

    def generate(self, hint):
        logging.info(f"Generating for hint: {hint}")
        max_len = int(len(hint) * 1.5 + 50)
        grams = random.randint(5, 10)
        result = self.pipe(hint, max_length=max_len, encoder_no_repeat_ngram_size=grams, num_beams=grams+1)[0]["generated_text"]
        logging.info(f"Generated for hint: {hint}: {result}")
        return result
