import torch
from tokenizers import Tokenizer
from model import *
from inference import get_translation
model = torch.load('opus_books_weights/tmodel_final_model_weights_en-it.pt')
src_tokenizer = Tokenizer.from_file('tokenizer_en.json')
tgt_tokenizer = Tokenizer.from_file('tokenizer_it.json')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seq_len = 350

text = input("Enter a sentence in English: ")

translation = get_translation(model, text, src_tokenizer, tgt_tokenizer, seq_len, device)

print("Translation: ", translation)