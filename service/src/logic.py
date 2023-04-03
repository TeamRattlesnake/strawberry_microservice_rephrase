import os
import re

import random
from tqdm import tqdm
import logging

import torch

from transformers import T5ForConditionalGeneration, T5Tokenizer

logging.basicConfig(format="%(asctime)s %(message)s", handlers=[logging.FileHandler(
    f"/home/logs/rephrase_log_model.txt", mode="w", encoding="UTF-8")], datefmt="%I:%M:%S %p", level=logging.INFO)


class NeuralNetwork:
    def __init__(self, group_id=0):
        self.DEVICE = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = "cointegrated/rut5-base-paraphraser"
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        self.group_id = group_id

    def generate(self, hint, do_sample=False):
        beams = 5,
        grams = 4,
        x = self.tokenizer(hint, return_tensors='pt',
                           padding=True).to(self.model.device)
        max_size = int(x.input_ids.shape[1] * 2.0 + 10)
        out = self.model.generate(**x, encoder_no_repeat_ngram_size=grams,
                                  num_beams=beams, max_length=max_size, do_sample=do_sample)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
