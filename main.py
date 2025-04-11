"""
Let a model generate code and get the code file.
"""
import sys
sys.path.append("libs")

import os

import argparse
import json

from utils.models import load_model, load_tokenizer, load_dataloader, inference, inference_greedy

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="codet5p-small"
)
parser.add_argument(
    "--is_peft",
    type=bool,
    default=False
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
    choices=["humaneval", "codet_humaneval", "mbpp_sanitized", "mbpp"]
)
parser.add_argument(
    "--k",
    type=int,
    default=5
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.95
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0.95
)
parser.add_argument(
    "--max_length",
    type=int,
    default=256
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=256
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8
)
parser.add_argument(
    "--mode",
    type=int,
    default=1
)
parser.add_argument(
    "--method",
    type=str,
    default="finetuned",
    choices=["finetuned", "persd", "standard", "origin", "r"]
)

args = parser.parse_args()

# load tokenizer
tokenizer = load_tokenizer(os.path.join("models", args.tokenizer), args.tokenizer, padding_left=True)

# load model
model_path = ""
if args.is_peft:
    model_path = os.path.join("models", args.model, args.method, args.dataset)
else:
    model_path = os.path.join("models", args.model)
model = load_model(model_path, is_peft=args.is_peft, is_train=False, is_decoder=args.is_decoder)

# dataloader
dataloder = load_dataloader(os.path.join("dataset", args.dataset + ".jsonl"), args.dataset, tokenizer, max_length=args.max_length, batch_size=args.batch_size, is_train=False)

# start inference
results = []
if args.mode == 1:
    results = inference(model, tokenizer, dataloder, max_new_tokens=args.max_new_tokens, k=args.k, temperature=args.temperature, top_p=args.top_p)
else:
    results = inference_greedy(model, tokenizer, dataloder, max_new_tokens=args.max_new_tokens)

# write results to file
# a file name consists of temperature, k and max_new_tokens
filename = "samples-" + str(args.temperature) + "-" + str(args.k) + "-" + str(args.max_new_tokens)
if args.mode != 1:
    filename = "samples-" + str(args.max_new_tokens)
output_filepath = os.path.join("samples", args.model, args.method, args.dataset, filename)
if not os.path.exists(os.path.join("samples", args.model, args.method, args.dataset)):
    os.makedirs(os.path.join("samples", args.model, args.method, args.dataset))

with open(output_filepath, "w") as f:
    for data in results:
        json_data = json.dumps(data)
        f.write(json_data + "\n")