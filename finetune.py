"""
Fine-tune the model directly
"""
import sys
sys.path.append("libs")

import os

import argparse
import json
from tqdm import tqdm

import torch
from transformers import get_scheduler

from utils.models import load_model, load_tokenizer, load_dataloader, fine_tune, CrossEntropyLoss

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
    "--batch_size",
    type=int,
    default=4
)
parser.add_argument(
    "--max_length",
    type=int,
    default=512
)
parser.add_argument(
    "--epochs",
    type=int,
    default=4
)
parser.add_argument(
    "--lr",
    type=float,
    default=5e-5
)
parser.add_argument(
    "--scheduler",
    type=str,
    default="linear",
    choices=["linear", "cosine"]
)
parser.add_argument(
    "--model_save_dir",
    type=str,
    default="finetuned"
)

args = parser.parse_args()

# load tokenizer
tokenizer = load_tokenizer(os.path.join("models", args.tokenizer), args.tokenizer, is_train=True)

# load model
model = load_model(os.path.join("models", args.model), is_peft=False, is_decoder=args.is_decoder, is_train=True)

# dataloader
dataloder = load_dataloader(os.path.join("dataset", args.dataset + ".jsonl"), args.dataset, tokenizer, max_length=args.max_length, batch_size=args.batch_size, is_train=True)

# fine-tune the model
num_warmup_steps = args.epochs // 10
num_training_steps = args.epochs * len(dataloder)

criterion = CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr
)
scheduler = get_scheduler(
    name=args.scheduler,
    optimizer=optimizer,
    num_warmup_steps=num_training_steps,
    num_training_steps=num_training_steps
)

model_save_dir = os.path.join("models", args.model, args.model_save_dir, args.dataset)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

fine_tune(
    model,
    dataloder,
    criterion,
    optimizer,
    scheduler,
    model_save_dir=model_save_dir,
    epochs=args.epochs,
    is_decoder=args.is_decoder
)