import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings('ignore')

# Load the pre-trained model
tokenizer_1 = AutoTokenizer.from_pretrained("kevinscaria/ate_tk-instruct-base-def-pos-neg-neut-combined")
model_1 = AutoModelForSeq2SeqLM.from_pretrained("kevinscaria/ate_tk-instruct-base-def-pos-neg-neut-combined")

bos_instruction = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
 keyboard
    Neutral example 1-
    input: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset.
    output: battery
    Neutral example 2-
    input: Night   Positive example 1-
    input: I charge it at night and skip taking the cord with me because of the good battery life.
    output: battery life
    Positive example 2-
    input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
    output: features, iChat, Photobooth, garage band
    Negative example 1-
    input: Speaking of the browser, it too has problems.
    output: browser
    Negative example 2-
    input: The keyboard is too slick.
    output: ly my computer defrags itself and runs a virus scan.
    output: virus scan
    Now complete the following example-
    input: """
# Read the data from a CSV file

def ATE_for_csv(data):
  aspect_terms = []
  for text in data['text']:
    delim_instruct = ''
    eos_instruct = ' \noutput:'
    tokenized_text_1 = tokenizer_1(bos_instruction + text + delim_instruct + eos_instruct, return_tensors="pt")
    output_1 = model_1.generate(tokenized_text_1.input_ids)
    aspect_terms.append(tokenizer_1.decode(output_1[0], skip_special_tokens=True))
  data['aspect_terms'] = aspect_terms
  return data


