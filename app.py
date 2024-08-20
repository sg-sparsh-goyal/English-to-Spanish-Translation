import streamlit as st
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformer import Transformer
import torch.nn.functional as F



english_file = 'dataset/english.txt'
spanish_file = 'dataset/spanish.txt'

START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'


english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '’',
                      '‘', ';', '₂',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      ':', '<', '=', '>', '?', '@',
                      '[', '\\', ']', '^', '_', '`',
                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                      'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                      'y', 'z',
                      'á', 'é', 'í', 'ó', 'ú', 'ñ', 'ü',
                      '¿', '¡',
                      'Á', 'É', 'Í', 'Ó', 'Ú', 'Ñ', 'Ü',
                      '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN,
                      'à', 'è', 'ì', 'ò', 'ù', 'À', 'È', 'Ì', 'Ò', 'Ù',
                      'â', 'ê', 'î', 'ô', 'û', 'Â', 'Ê', 'Î', 'Ô', 'Û',
                      'ä', 'ë', 'ï', 'ö', 'ü', 'Ä', 'Ë', 'Ï', 'Ö',
                      'ã', 'õ', 'Ã', 'Õ',
                      'ā', 'ē', 'ī', 'ō', 'ū', 'Ā', 'Ē', 'Ī', 'Ō', 'Ū',
                      'ą', 'ę', 'į', 'ǫ', 'ų', 'Ą', 'Ę', 'Į', 'Ǫ', 'Ų',
                      'ç', 'Ç', 'ş', 'Ş', 'ğ', 'Ğ', 'ń', 'Ń', 'ś', 'Ś', 'ź', 'Ź', 'ż', 'Ż',
                      'č', 'Č', 'ć', 'Ć', 'đ', 'Đ', 'ł', 'Ł', 'ř', 'Ř', 'š', 'Š', 'ť', 'Ť',
                      'ý', 'ÿ', 'Ý', 'Ÿ', 'ž', 'Ž', 'ß', 'œ', 'Œ', 'æ', 'Æ', 'å', 'Å', 'ø', 'Ø', 'å', 'Å',
                      'æ', 'Æ', 'œ', 'Œ']


spanish_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '’',
                      '‘', ';', '₂',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      ':', '<', '=', '>', '?', '@',
                      '[', '\\', ']', '^', '_', '`',
                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                      'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                      'y', 'z',
                      'á', 'é', 'í', 'ó', 'ú', 'ñ', 'ü',
                      '¿', '¡',
                      'Á', 'É', 'Í', 'Ó', 'Ú', 'Ñ', 'Ü',
                      '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN,
                      'à', 'è', 'ì', 'ò', 'ù', 'À', 'È', 'Ì', 'Ò', 'Ù',
                      'â', 'ê', 'î', 'ô', 'û', 'Â', 'Ê', 'Î', 'Ô', 'Û',
                      'ä', 'ë', 'ï', 'ö', 'ü', 'Ä', 'Ë', 'Ï', 'Ö',
                      'ã', 'õ', 'Ã', 'Õ',
                      'ā', 'ē', 'ī', 'ō', 'ū', 'Ā', 'Ē', 'Ī', 'Ō', 'Ū',
                      'ą', 'ę', 'į', 'ǫ', 'ų', 'Ą', 'Ę', 'Į', 'Ǫ', 'Ų',
                      'ç', 'Ç', 'ş', 'Ş', 'ğ', 'Ğ', 'ń', 'Ń', 'ś', 'Ś', 'ź', 'Ź', 'ż', 'Ż',
                      'č', 'Č', 'ć', 'Ć', 'đ', 'Đ', 'ł', 'Ł', 'ř', 'Ř', 'š', 'Š', 'ť', 'Ť',
                      'ý', 'ÿ', 'Ý', 'Ÿ', 'ž', 'Ž', 'ß', 'œ', 'Œ', 'æ', 'Æ', 'å', 'Å', 'ø', 'Ø', 'å', 'Å',
                      'æ', 'Æ', 'œ', 'Œ']

index_to_english = {k: v for k, v in enumerate(english_vocabulary)}
english_to_index = {v: k for k, v in enumerate(english_vocabulary)}
index_to_spanish = {k: v for k, v in enumerate(spanish_vocabulary)}
spanish_to_index = {v: k for k, v in enumerate(spanish_vocabulary)}

d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 200
es_vocab_size = len(spanish_vocabulary)  # Ensure you have spanish_vocabulary defined elsewhere

# Instantiate the Transformer model
transformer = Transformer(d_model,
                          ffn_hidden,
                          num_heads,
                          drop_prob,
                          num_layers,
                          max_sequence_length,
                          es_vocab_size,
                          english_to_index,  # Ensure english_to_index is defined elsewhere
                          spanish_to_index,  # Ensure spanish_to_index is defined elsewhere
                          START_TOKEN,
                          END_TOKEN,
                          PADDING_TOKEN)


class TextDataset(Dataset):
    def __init__(self, english_sentences, spanish_sentences):
        self.english_sentences = english_sentences
        self.spanish_sentences = spanish_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.spanish_sentences[idx]


device = "cpu"

NEG_INFTY = -1e9


def create_masks(eng_batch, kn_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)

    for idx in range(num_sentences):
        eng_sentence_length, kn_sentence_length = len(eng_batch[idx]), len(kn_batch[idx])
        eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
        kn_chars_to_padding_mask = np.arange(kn_sentence_length + 1, max_sequence_length)
        encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
        encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


def preprocess_text(text):
    # Remove newline characters and extra spaces
    return text.replace('\n', ' ').strip()


def translate(eng_sentence):
    transformer.load_state_dict(torch.load("englishTOspanish.pt", map_location=torch.device('cpu')))
    transformer.eval()

    # Preprocess the input sentence
    eng_sentence = preprocess_text(eng_sentence)
    eng_sentence = (eng_sentence,)
    es_sentence = ("",)

    for word_counter in range(max_sequence_length):
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
            eng_sentence, es_sentence)

        predictions = transformer(eng_sentence,
                                  es_sentence,
                                  encoder_self_attention_mask.to(device),
                                  decoder_self_attention_mask.to(device),
                                  decoder_cross_attention_mask.to(device),
                                  enc_start_token=False,
                                  enc_end_token=False,
                                  dec_start_token=True,
                                  dec_end_token=False)

        next_token_prob_distribution = predictions[0][word_counter]

        # Apply softmax temperature to control randomness
        temperature = 0.7
        next_token_prob_distribution = F.softmax(next_token_prob_distribution / temperature, dim=-1)

        # Add repetition penalty
        for token in es_sentence[0]:
            next_token_prob_distribution[spanish_to_index[token]] *= 0.1

        next_token_index = torch.argmax(next_token_prob_distribution).item()
        next_token = index_to_spanish.get(next_token_index, END_TOKEN)  # Handle missing index gracefully

        # Stop if we reach the end token
        if next_token == END_TOKEN:
            break

        # Avoid repeating tokens
        if len(es_sentence[0]) == 0 or next_token != es_sentence[0][-len(next_token):]:
            es_sentence = (es_sentence[0] + next_token,)

    # Clean the output
    return es_sentence[0].replace(END_TOKEN, '').strip()


# Streamlit UI
st.title("seq2seq Machine Translation")
st.write("Translate English to Spanish")
st.write("\n")
st.write("Some example sentences:")
st.write("i'm happy to see you here")
st.write("i have nothing to do with it")
st.write("what did you say yesterday?")
st.write("\n")

input_text = st.text_area("Enter English text:")

if st.button("Translate"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        translated_text = translate(input_text)
        st.write("Your Text (English):")
        st.title(input_text)
        st.write("Translated Text (Spanish):")
        st.title(translated_text)