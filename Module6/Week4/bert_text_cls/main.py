import numpy as np
import evaluate

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
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

processed_dataset = ds.map(
    preprocess_function, batched=True, desc="Runiing tokenizer on dataset"
)

num_labels = 2
config = AutoConfig.from_pretrained(
    model_name, num_labels=num_labels, finetuning_task="text-classification"
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, config=config
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    result = metric.compute(predictions=predictions, references=labels)
    return result

training_args = TrainingArguments(
    output_dir="save_model",
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["valid"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)