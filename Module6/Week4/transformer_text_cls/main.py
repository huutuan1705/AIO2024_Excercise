import os
import torch
import torch.optim as optim

from datasets import load_dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from encoder import TransformerEncoderCls

import utils

ds = load_dataset('thainq107/ntc-scv')
tokenizer = get_tokenizer("basic_english")
vocab_size = 10000
max_length = 100
embed_dim = 200
num_layers = 2
num_heads = 4
ff_dim = 128
dropout = 0.1

vocabulary = build_vocab_from_iterator(
    utils.yeild_tokens(ds['train']['preprocessed_sentence'], tokenizer),
    max_tokens=vocab_size,
    specials=["<pad>", "<unk>"]
)
vocabulary.set_default_index(vocabulary["unk"])

train_dataset = utils.prepare_dataset(ds['train'], vocabulary, tokenizer)
train_dataset = to_map_style_dataset(train_dataset)

valid_dataset = utils.prepare_dataset(ds['valid'], vocabulary, tokenizer)
valid_dataset = to_map_style_dataset(valid_dataset)

test_dataset = utils.prepare_dataset(ds['test'], vocabulary, tokenizer)
test_dataset = to_map_style_dataset(test_dataset)

batch_size=128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=utils.collate_batch)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=utils.collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=utils.collate_batch)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerEncoderCls(
    vocab_size, max_length, num_layers, embed_dim, num_heads, ff_dim, dropout, device
)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.00005)

num_epochs = 50
save_model = './model'
os.makedirs(save_model, exist_ok=True)
model_name = 'model'

model, metrics = utils.train(
    model, model_name, save_model, optimizer, criterion, train_dataloader, valid_dataloader,
    num_epochs, device
)

utils.plot_result(num_epochs, metrics["train_accuracy"], metrics["valid_accuracy"],
                  metrics["train_loss"], metrics["valid_loss"])
