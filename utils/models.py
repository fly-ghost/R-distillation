"""
模型相关代码
"""
from tqdm import tqdm
from functools import partial

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, TaskType, LoraConfig, AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM

from utils.data import CustomDataset
from utils.preprocess import tokenize, collate_fn, collate_fn_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, is_peft=False, is_decoder=False, is_train=False, device_ids=[0]):
    """
    加载模型, 要区分是否是LoRA加载, 是否是仅解码器架构
    """
    model = None

    # is peft or not
    if is_peft is True:
        if is_decoder is True:
            model = AutoPeftModelForCausalLM.from_pretrained(model_path, is_decoder=True, trust_remote_code=True)
        else:
            model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
    else:
        if is_decoder is True:
            model = AutoModelForCausalLM.from_pretrained(model_path, is_decoder=True, trust_remote_code=True)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)

    # is train or not
    if is_train is True:
        task_type = TaskType.CAUSAL_LM
        if is_decoder is False:
            task_type = TaskType.SEQ_2_SEQ_LM
        peft_config = LoraConfig(
            task_type=task_type, inference_mode=False, lora_alpha=16, lora_dropout=0.1, r=8, bias="none"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    # 多卡环境需要分配device
    if len(device_ids) > 1:
        device = torch.device("cuda:{0}".format(device_ids[0]))
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        device = torch.device("cuda:0")
    model = model.to(device)

    return model

def load_tokenizer(tokenizer_path, tokenizer_name, padding_left=True):
    """
    加载分词器, 默认左padding
    """
    padding_side = "left"
    if padding_left is False:
        padding_side = "right"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side=padding_side)
    if tokenizer_name in ["codegen-mono-small", "codegen-mono-base", "codegen-multi-small", "codegen-multi-base"]:
        tokenizer.add_special_tokens({"pad_token":'<pad>'})

    return tokenizer

def load_dataset(dataset_path, dataset_name, is_train=False):
    """
    加载数据集
    """
    dataset = CustomDataset(dataset_path, dataset_name, is_train=is_train)
    return dataset

def load_dataloader(dataset_path, dataset_name, tokenizer, max_length=256, batch_size=8, is_train=False, is_decoder=False):
    """
    设置DataLoader
    """
    dataset = CustomDataset(dataset_path, dataset_name, is_train=is_train)
    tokenize_preprocessing = partial(
        tokenize,
        tokenizer=tokenizer,
        max_length=max_length,
        is_train=is_train,
        is_decoder=is_decoder
    )
    dataset.map(tokenize_preprocessing)
    dataloader = None
    if is_train is True:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_inference
        )
    return dataloader

def inference_greedy(model, tokenizer, dataloader, max_new_tokens=256):
    """
    贪婪解码模式生成代码
    """
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=max_new_tokens
            )

            result = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False
            )
            task_ids = batch["task_ids"]
            for i in range(len(result)):
                data = {
                    "task_id": task_ids[i],
                    "completion": result[i]
                }
                results.append(data)

    return results

def inference(model, tokenizer, dataloader, max_new_tokens=256, k=10, temperature=0.95, top_p=0.95):
    """
    beam search模型生成代码
    """
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_beams=k,
                num_return_sequences=k,
                temperature=temperature,
                top_p=top_p
            )

            result = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False
            )
            task_ids = batch["task_ids"]
            for i in range(len(result)):
                data = {
                    "task_id": task_ids[i//k],
                    "completion": result[i]
                }
                results.append(data)

    return results

class DistillationLoss(torch.nn.Module):
    """
    标准蒸馏损失函数, 交叉熵 + KL
    """
    def __init__(self, T=1.0, a=0.5):
        super(DistillationLoss, self).__init__()
        self.first_loss_function = torch.nn.CrossEntropyLoss(reduction="mean")
        self.second_loss_function = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")
        self.T = T
        self.a = a

    def get_T(self):
        return self.T

    def set_T(self, T):
        self.T = T

    def forward(self, logits, labels, teacher_logits):
        teacher_probs = F.log_softmax(teacher_logits / self.T, dim=-1)
        student_log_probs = F.log_softmax(logits, dim=-1)
        first_loss = self.first_loss_function(logits.permute(0, 2, 1), labels)
        second_loss = self.second_loss_function(student_log_probs, teacher_probs)
        total_loss = self.a * first_loss + (1 - self.a) * second_loss
        return total_loss

class CrossEntropyLoss(torch.nn.Module):
    """
    交叉熵损失函数
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, logits, labels):
        loss = self.loss_function(logits.permute(0, 2, 1), labels)
        return loss  

class RLoss(torch.nn.Module):
    """
    RLHF损失函数, 交叉熵 + 强化学习的奖励
    """
    def __init__(self, agent, a=0.8):
        super(RLoss, self).__init__()
        self.c_loss_function = torch.nn.CrossEntropyLoss(reduction="mean")
        self.r_loss_function = agent
        self.a = a

    def forward(self, logits, labels):
        c_loss = self.c_loss_function(logits.permute(0, 2, 1), labels)
        r_loss = self.r_loss_function.get_loss()
        loss = self.a * c_loss + (1 - self.a) * r_loss
        return loss

def fine_tune(model, dataloader, criterion, optimizer, scheduler, model_save_dir, epochs=10, is_decoder=True):
    """
    直接微调, 只保存loss最小的结果
    """
    for i in range(1, epochs + 1):
        print("start epoch: {0}".format(i))
        total_loss = 0.0
        min_loss = 0.0
        model.train()
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            output = None
            if is_decoder is True:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    labels=input_ids
                )
            else:
                output = model(
                    input_ids=input_ids,
                    decoder_input_ids=labels,
                    attention_mask=attention_masks
                )
            logits = output.logits

            loss = criterion(logits, labels)
            loss.backward()

            total_loss += loss.item()
            optimizer.step()
            scheduler.step()

        if min_loss == 0.0 or total_loss < min_loss:
            model.save_pretrained(model_save_dir)
        print("\tloss: {0}".format(total_loss / len(dataloader)))

def persd(model, dataloader, criterion, optimizer, scheduler, model_save_dir, agent, epochs=20, is_decoder=False):
    """
    个性化蒸馏
    """
    for i in range(1, epochs+1):
        print("start epoch: {0}".format(i))
        total_loss = 0.0
        model.train()
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            output = None
            if is_decoder is True:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_masks
                )
            else:
                output = model(
                    input_ids=input_ids,
                    decoder_input_ids=labels,
                    attention_mask=attention_masks
                )
            logits = output.logits

            loss = criterion(logits, labels)
            loss.backward()

            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
        
        # 一轮迭代后, 进行评估
        agent.predict()
        model.save_pretrained(model_save_dir)
        print("\tloss: {0}".format(total_loss / len(dataloader)))

def r_distillation(model, dataloader, criterion, optimizer, scheduler, model_save_dir, agent, epochs=20, is_decoder=False, step=1):
    """
    RLHF蒸馏, 保留奖励值最高(奖励loss最低)的结果
    """
    for i in range(1, epochs+1):
        print("start epoch: {0}".format(i))
        total_loss = 0.0
        model.train()
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            output = None
            if is_decoder is True:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_masks
                )
            else:
                output = model(
                    input_ids=input_ids,
                    decoder_input_ids=labels,
                    attention_mask=attention_masks
                )
            logits = output.logits

            loss = criterion(logits, labels)
            loss.backward()

            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
        agent.show_details()
        agent.reset()
        model.save_pretrained(model_save_dir)
        print("\tloss: {0}".format(total_loss / len(dataloader)))

def standard_distillation(student_model, teacher_model, dataloader, criterion, optimizer, scheduler, model_save_dir, epochs=10, is_decoder=False, device_ids=[0], teacher_device_ids=[0]):
    """
    标准蒸馏, 需要同时加载学生模型和教师模型
    """
    teacher_model.eval()
    for i in range(1, epochs+1):
        print("start epoch: {0}".format(i))
        total_loss = 0.0
        min_loss = 0.0
        student_model.train()
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device_ids[0])
            attention_masks = batch["attention_masks"].to(device_ids[0])
            labels = batch["labels"].to(device_ids[0])

            optimizer.zero_grad()

            student_output = None
            if is_decoder is True:
                student_output = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_masks
                )
            else:
                student_output = student_model(
                    input_ids=input_ids,
                    decoder_input_ids=labels,
                    attention_mask=attention_masks
                )
            student_logits = student_output.logits

            input_ids = input_ids.to(teacher_device_ids[0])
            attention_masks = attention_masks.to(teacher_device_ids[0])
            labels = labels.to(teacher_device_ids[0])
            teacher_output = None
            with torch.no_grad():
                if is_decoder is True:
                    teacher_output = teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_masks
                    )
                else:
                    teacher_output = teacher_model(
                        input_ids=input_ids,
                        decoder_input_ids=labels,
                        attention_mask=attention_masks
                    )
            teacher_logits = teacher_output.logits


            student_logits = student_logits.to(teacher_device_ids[0])
            loss = criterion(student_logits, labels, teacher_logits)

            total_loss += loss.item()
            loss.backward()
            
            optimizer.step()
            scheduler.step()

        module = student_model
        if len(device_ids) > 1:
            module = student_model.module
        if min_loss == 0.0 or total_loss < min_loss:
            module.save_pretrained(model_save_dir)
        print("\tloss: {0}".format(total_loss / len(dataloader)))