"""
个性化蒸馏, 本质上是数据增强
"""
import sys
sys.path.append("libs")

import os
import random

import numpy as np

import argparse
import json
from tqdm import tqdm

import torch
from transformers import get_scheduler

from utils.models import load_model, load_tokenizer, load_dataloader, load_dataset, persd, CrossEntropyLoss
from reinforcement import Agent

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="codet5p-small"
)
parser.add_argument(
    "--is_decoder",
    type=bool,
    default=False
)
parser.add_argument(
    "--tokenizer",
    type=str,
    default="codet5p-small"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="humaneval",
)
parser.add_argument(
    "--validation_dataset",
    type=str,
    default="humaneval",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8
)
parser.add_argument(
    "--max_length",
    type=int,
    default=256
)
parser.add_argument(
    "--epochs",
    type=int,
    default=5
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-4
)
parser.add_argument(
    "--scheduler",
    type=str,
    default="linear",
    choices=["linear", "cosine", "cosine_with_restarts"]
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=256
)
parser.add_argument(
    "--model_save_dir",
    type=str,
    default="persd"
)

args = parser.parse_args()

# load tokenizer
tokenizer = load_tokenizer(os.path.join("models", args.tokenizer), args.tokenizer, padding_left=True)

# load model
model = load_model(os.path.join("models", args.model), is_peft=False, is_decoder=args.is_decoder, is_train=True)

# dataloader
dataloader = load_dataloader(os.path.join("dataset", args.dataset + ".jsonl"), args.dataset, tokenizer, max_length=args.max_length, batch_size=args.batch_size, is_train=True)

# dataset for agent
validation_dataset = load_dataset(os.path.join("dataset", args.validation_dataset + ".jsonl"), args.validation_dataset, is_train=True)

# dataloader for agent
validation_tokenizer = load_tokenizer(os.path.join("models", args.tokenizer), args.tokenizer, padding_left=True)
validation_dataloader = load_dataloader(os.path.join("dataset", args.validation_dataset + ".jsonl"), args.validation_dataset, validation_tokenizer, max_length=args.max_length, batch_size=args.batch_size, is_train=False, is_decoder=args.is_decoder)

# fine-tune the model
num_warmup_steps = args.epochs // 10
num_training_steps = args.epochs * len(dataloader)

criterion = CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr
)
scheduler = get_scheduler(
    name=args.scheduler,
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

model_save_dir = os.path.join("models", args.model, args.model_save_dir, args.dataset)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

agent = Agent(model, validation_tokenizer, validation_dataset, validation_dataloader, args)

persd(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    model_save_dir,
    agent,
    epochs=args.epochs,
    is_decoder=args.is_decoder
)