import torch
from tokenizers import Tokenizer
from model import *

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def get_translation(model, text, tokenizer_src, tokenizer_tgt, seq_len, device):
    model.eval()

    sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    with torch.no_grad():
        enc_input_tokens = tokenizer_src.encode(text).ids
        enc_num_padding_tokens = seq_len - len(enc_input_tokens) - 2

        encoder_input = torch.cat(
            [
                sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                eos_token,
                torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        encoder_input = encoder_input.unsqueeze(0) # (1, 1, seq_len)
        encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int() # (b, 1, 1, seq_len)

        # check that the batch size is 1
        assert encoder_input.size(
            0) == 1, "Batch size must be 1 for validation"

        model_out = greedy_decode(model, encoder_input.to(device), encoder_mask.to(device), tokenizer_tgt, seq_len, device)

        model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        return model_out_text

# text = "Hello, my name is John. I am a student."

# print(run_validation(model, text, src_tokenizer, tgt_tokenizer, 350, 'cuda'))
