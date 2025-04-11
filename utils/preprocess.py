import torch

def tokenize(data, tokenizer, max_length=256, is_train=False, is_decoder=False):
    # 自回归模型的prompt是prompt+label
    result = None
    if is_decoder is True:
        if is_train is True:
            result = tokenizer(
                data["prompt"] + data["label"],
                max_length=max_length,
                padding="max_length",
                truncation=True
            )
        else:
            result = tokenizer(
                data["prompt"],
                max_length=max_length,
                padding="max_length",
                truncation=True
            )
    else:
        result = tokenizer(
            data["prompt"],
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
    if is_train is True:
        # 无论自回归模型还是编码器-解码器模型，label都是prompt+label，用于交叉熵损失函数
        result["label"] = tokenizer(
            data["prompt"] + data["label"],
            max_length=max_length,
            padding="max_length",
            truncation=True
        )["input_ids"]
        # 需要将label左移一位, 因为logits表示的是下一个token的预测
        result["label"] = [x for x in result["label"][1:]]
        result["label"].append(tokenizer.pad_token_id)

    result["task_id"] = data["task_id"]

    return result

def collate_fn(lines):
    task_ids = []
    input_ids = []
    labels = []
    attention_masks = []
    for row in lines:
        task_ids.append(row["task_id"])
        input_ids.append(row["input_ids"])
        labels.append(row["label"])
        attention_masks.append(row["attention_mask"])
    
    return {
        "task_ids": task_ids,
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_masks": torch.tensor(attention_masks)
    }

def collate_fn_inference(lines):
    task_ids = []
    input_ids = []
    attention_masks = []
    for row in lines:
        task_ids.append(row["task_id"])
        input_ids.append(row["input_ids"])
        attention_masks.append(row["attention_mask"])
    
    return {
        "task_ids": task_ids,
        "input_ids": torch.tensor(input_ids),
        "attention_masks": torch.tensor(attention_masks)
    }