import math
import os

import torch
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from model import VisionTransformerCls
import utils

# !gdown 1vSevps_hV5zhVf6aWuN8X7dd-qSAIgcc
# !unzip ./flower_photos.zip

# Load data
data_path = "./flower_photos"
dataset = ImageFolder(root=data_path)
num_samples = len(dataset)
classes = dataset.classes
num_classes = len(dataset.classes)

# split
TRAIN_RATIO, VALID_RATIO = 0.8, 0.1
n_train_examples = int(num_samples*TRAIN_RATIO)
n_valid_examples = int(num_samples*VALID_RATIO)
n_test_examples = num_samples - n_train_examples - n_valid_examples
train_dataset, valid_dataset, test_dataset = random_split(
    dataset, [n_train_examples, n_valid_examples, n_test_examples]
)

# PREPROCESSING
IMG_SIZE = 224
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# apply
train_dataset.dataset.transform = train_transform
valid_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = train_transform

# Loader data
BATCH_SIZE = 512
train_loader = DataLoader(
    train_dataset, shuffle=True, batch_size=BATCH_SIZE
)
val_loader = DataLoader(
    valid_dataset, shuffle=True, batch_size=BATCH_SIZE
)
test_loader = DataLoader(
    test_dataset, shuffle=True, batch_size=BATCH_SIZE
)

# TRAINING
embed_dim = 512
num_heads = 4
ff_dim = 128
dropout = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformerCls(image_size=IMG_SIZE, embed_dim=embed_dim, num_heads=num_heads, 
            ff_dim=ff_dim, dropout=dropout, num_classes=num_classes, device=device)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

num_epochs = 100
save_model = './vit_flosers'
os.makedirs(save_model, exist_ok=True)
model_name = 'vit_flowers'

model, metrics = utils.train(
    model, model_name, save_model, optimizer, criterion, train_loader, val_loader,
    num_epochs, device
)