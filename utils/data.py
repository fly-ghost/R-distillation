import json

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset_path, dataset_name, is_train=False) -> None:
        super().__init__()
        self.dataset = []
        # 不同的数据集需要不同的处理
        if dataset_name.startswith("humaneval"):
            with open(dataset_path, "r") as f:
                for line in f:
                    json_obj = json.loads(line)
                    data = {
                        "task_id": json_obj["task_id"],
                        "prompt": json_obj["prompt"]
                    }
                    if is_train is True:
                        data["label"] = json_obj["canonical_solution"]
                        data["test"] = json_obj["test"]
                        data["entry_point"] = json_obj["entry_point"]
                    self.dataset.append(data)
        elif dataset_name.startswith("codet_humaneval"):
            with open(dataset_path, "r") as f:
                for line in f:
                    json_obj = json.loads(line)
                    data = {
                        "task_id": json_obj["task_id"],
                        "prompt": json_obj["prompt"]
                    }
                    if is_train is True:
                        data["label"] = json_obj["canonical_solution"]
                        data["test"] = json_obj["test"]
                        data["entry_point"] = json_obj["entry_point"]
                    self.dataset.append(data)
        elif dataset_name.startswith("mbpp_sanitized"):
            with open(dataset_path, "r") as f:
                for line in f:
                    json_obj = json.loads(line)
                    data = {
                        "task_id": json_obj["task_id"],
                        "prompt": json_obj["prompt"]
                    }
                    if is_train is True:
                        data["label"] = json_obj["canonical_solution"]
                        data["test"] = json_obj["test"]
                        data["entry_point"] = json_obj["entry_point"]
                    self.dataset.append(data)
        elif dataset_name == "mbpp":
            with open(dataset_path, "r") as f:
                for line in f:
                    json_obj = json.loads(line)
                    data = {
                        "task_id": json_obj["task_id"],
                        "prompt": json_obj["text"]
                    }
                    if is_train is True:
                        data["label"] = json_obj["code"]
                        data["test"] = "\n".join(json_obj["test_list"])
                    self.dataset.append(data)
        elif dataset_name.startswith("mbpp_validation"):
            with open(dataset_path, "r") as f:
                for line in f:
                    json_obj = json.loads(line)
                    data = {
                        "task_id": json_obj["task_id"],
                        "prompt": json_obj["prompt"]
                    }
                    if is_train is True:
                        data["label"] = json_obj["canonical_solution"]
                        data["test"] = json_obj["test"]
                        data["entry_point"] = json_obj["entry_point"]
                        data["prompt_code"] = json_obj["prompt_code"]
                    self.dataset.append(data)
        elif dataset_name.startswith("apps"):
            with open(dataset_path, "r") as f:
                for line in f:
                    json_obj = json.loads(line)
                    data = {
                        "task_id": json_obj["task_id"],
                        "prompt": json_obj["prompt"]
                    }
                    if is_train is True:
                        data["label"] = json_obj["canonical_solution"]
                    self.dataset.append(data)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def map(self, func):
        for i in range(len(self.dataset)):
            self.dataset[i] = func(self.dataset[i])
