
from transformers import AutoTokenizer
from datasets import load_dataset

ds = load_dataset('thainq07/ntc-scv')
model_name = "distilbert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
max_seq_length = 100
max_seq_length = min(max_seq_length, tokenizer.model_max_length)

def preprocess_function(examples):
    result = tokenizer(
        examples['preprocessed_sentence'],
        padding="max_length",
        truncation=True
    )
    result["label"] = examples['label']

    return result

precessed_dataset = ds.map(
    preprocess_function, batched=True, desc="Runiing tokenizer on dataset"
)