import torch
from tokenizers import Tokenizer
from model import *
from inference import get_translation
import gradio as gr

device = torch.device('cpu') # setting device to cpu to run of huggingface spaces free tier

model = torch.load('opus_books_weights/tmodel_final_model_weights_en-it.pt').to(device)
src_tokenizer = Tokenizer.from_file('tokenizer_en.json')
tgt_tokenizer = Tokenizer.from_file('tokenizer_it.json')

seq_len = 350

def translate(text):
    translation = get_translation(model, text, src_tokenizer, tgt_tokenizer, seq_len, device)
    return translation

title = "English to Italian Translator"
description = "A simple English to Italian translator using a Transformer model trained on the OPUS Books dataset."

gr.Interface(fn=translate, inputs="text", outputs="text", title=title, description=description).launch()