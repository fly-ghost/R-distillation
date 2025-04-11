import torch
import torch.nn.functional as F
from tqdm import tqdm

import contextlib
import signal
import io

class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False

class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, model, dataloader, tokenizer, dataset, args):
        self.model = model
        self.dataloader = list(dataloader)
        self.tokenizer = tokenizer
        self.dataset = dataset

        self.args = args

        self.n = len(self.dataset)
        self.m = args.max_new_tokens
        self.t_len = tokenizer.vocab_size
        # 记录dataloader的第几个元素, 用于循环, 因为验证集数量肯定小于训练集
        self.i = 0

        # task_id字典, 用于定位某个特定的验证记录
        self.test_map = dict()
        for j in range(self.n):
            self.test_map[self.dataset[j]["task_id"]] = self.dataset[j]
        
        # 奖励值
        self.FAILED = -1.0
        self.COMPILED = -0.5
        self.PASS_ONE = 0.3
        self.PASS_ALL = 1.0
        # 总奖励值
        self.v_all = 0.0
        # 当前轮次的正确率
        self.total_correct = 0.0
        # 当前轮次的编译通过率
        self.total_compiled = 0.0
        # 一共使用了多少个测试数据
        self.num = 0

    def reset(self):
        # 在一轮迭代后重置计数
        self.total_correct = 0.0
        self.total_compiled = 0.0
        self.v_all = 0.0
        self.num = 0
        # 让指针重新指向0
        self.i = 0

    def get_loss(self):
        # 按顺序取dataloader中的元素
        batch = self.dataloader[self.i]
        self.set_next()
        # 使用model进行前向传播得到logits, 为了能与模型真实生成的代码结合, 这里是left-padding
        # 前向传播过程
        input_ids = batch["input_ids"].to(device)
        attention_masks = batch["attention_masks"].to(device)

        output = None
        if self.args.is_decoder:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_masks
            )
        else:
            output = self.model(
                input_ids=input_ids,
                decoder_input_ids=input_ids,
                attention_mask=attention_masks
            )
        # logits需要是require_grad=True的
        logits = output.logits

        # 构建强化学习矩阵, shape为(batch_size, m, len(tokenizer))
        table = torch.zeros((len(logits), self.m, logits.shape[-1]), dtype=float).to(device)

        # 让模型真实生成代码
        result = []
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=self.m
            )

            result = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

        task_ids = batch["task_ids"]
        # 判断代码的正确性, 并计算奖励
        for j in range(len(result)):
            task_id = task_ids[j]
            completion = result[j]
            data = self.test_map[task_id]
            # 过滤completion
            completion_clean = self.truncate_main(completion)
            completion_clean = self.truncate(completion_clean, is_decoder=self.args.is_decoder)
            completion_clean = self.clean_print(completion_clean, is_decoder=self.args.is_decoder)
            completion_clean = self.clean_last_line(completion_clean)
            
            # 获取这个测试样本的所有单元测试用例
            test_list = self.get_test_list(data["test"])
            v = self.FAILED
            is_pass_all_tests = True
            for test in test_list:
                check_globals = {}
                check_program = None
                if self.args.is_decoder is False:
                    check_program = (
                        data["prompt"] + completion_clean + "\n" +
                        test + "\n" +
                        f"check({data['entry_point']})"
                    )
                else:
                    check_program = (
                        completion_clean + "\n" +
                        test + "\n" +
                        f"check({data['entry_point']})"
                        )
                with swallow_io():
                    with time_limit(3.0):
                        try:
                            c = compile(check_program, "<dynamic>", "exec")
                            if c is not None:
                                v = max(self.COMPILED, v)
                            exec(check_program, check_globals)
                            v = max(self.PASS_ONE, v)
                        except Exception:
                            is_pass_all_tests = False
            if is_pass_all_tests is True:
                v = self.PASS_ALL
            # 更新强化学习矩阵
            for k in range(self.m):
                table[j][k][generated_ids[j][k]-1] = v
            # 更新总奖励
            self.v_all += v
            if v != self.FAILED:
                self.total_compiled += 1
            if v == self.PASS_ALL:
                self.total_correct += 1
            self.num += 1

        # 对logits进行softmax, 得到概率分布, 在进行log
        logits = torch.log_softmax(logits, dim=-1)
        # 将logits和table相乘
        loss = logits * table
        # 对每一行求和, 再求平均
        loss = torch.sum(loss, dim=-1)
        return torch.mean(loss)

    def set_next(self):
        self.i = (self.i + 1) % len(self.dataloader)

    def truncate(self, completion, is_decoder=False):
        # 去除掉最后一个def，因为最后一个def大概率不完整，并且保证至少有一个def
        group = completion.rsplit("def", 1)
        if group[0].find("def") != -1:
            return group[0]
        else:
            return completion

    def clean_print(self, completion, is_decoder=False):
        # 去除掉print的部分，但仅解码器的print可能在prompt中
        if is_decoder is False:
            index = completion.rfind("print")
            if index != -1:
                completion = completion[:index]
        else:
            index_def = completion.find("def")
            index = completion.rfind("print", index_def)
            if index != -1:
                completion = completion[:index]
        return completion

    def truncate_main(self, completion):
        index = completion.find("if __name")
        if index != -1:
            return completion[:index]
        return completion

    def clean_last_line(self, completion):
        # 最后一行可能不完整, 或者多余, 会导致代码错, 去除
        return completion[:completion.rfind('\n')]

    def get_test_list(self, test):
        # 对test函数进行分解, 得到多个test
        lines = test.split("\n")
        # 从开始到函数定义, 甚至到第一个assert之前的部分都要
        j = 0
        while j < len(lines):
            if lines[j].strip().startswith("assert"):
                break
            j = j + 1
        if j >= len(lines):
            return [test]
        function_before = "\n".join(lines[:j])
        test_list = []
        # assert之前可能有一些变量的定义, 所以直到下一个assert之前都要加上
        start_line = j
        for i in range(j, len(lines)):
            if lines[i].strip().startswith("assert"):
                test_list.append(function_before + "\n" + "\n".join(lines[start_line:i+1]) + "\n")
                start_line = i + 1
        return test_list

    def show_details(self):
        if self.num == 0:
            print("\tPlease Start Traninig")
        else:
            print("\tSuccess Rate:{0}".format(self.total_correct / self.num))
            print("\tCompile Rate:{0}".format(self.total_compiled / self.num))
            print("\tMean Reward {0}".format(self.v_all / self.num))

    def predict(self):
        # 生成代码并进行评估
        print("\tstart predict...")
        total_correct = 0.0
        total_complied = 0.0
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_masks = batch["attention_masks"].to(device)
                
                # 让模型真实生成代码
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=self.m
                )

                result = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )

                task_ids = batch["task_ids"]
                # 判断代码的正确性, 并计算奖励
                for j in range(len(result)):
                    task_id = task_ids[j]
                    completion = result[j]
                    data = self.test_map[task_id]
                    # 过滤completion
                    completion_clean = self.truncate_main(completion)
                    completion_clean = self.truncate(completion_clean, is_decoder=self.args.is_decoder)
                    completion_clean = self.clean_print(completion_clean, is_decoder=self.args.is_decoder)
                    completion_clean = self.clean_last_line(completion_clean)
                    
                    test_list = self.get_test_list(data["test"])
                    v = self.FAILED
                    is_pass_all_tests = True
                    for test in test_list:
                        check_globals = {}
                        check_program = None
                        if self.args.is_decoder is False:
                            check_program = (
                                data["prompt"] + completion_clean + "\n" +
                                test + "\n" +
                                f"check({data['entry_point']})"
                            )
                        else:
                            check_program = (
                                completion_clean + "\n" +
                                test + "\n" +
                                f"check({data['entry_point']})"
                                )
                        with swallow_io():
                            with time_limit(3.0):
                                try:
                                    c = compile(check_program, "<dynamic>", "exec")
                                    if c is not None:
                                        v = max(self.COMPILED, v)
                                    exec(check_program, check_globals)
                                    v = max(self.PASS_ONE, v)
                                except Exception:
                                    is_pass_all_tests = False
                    if is_pass_all_tests is True:
                        v = self.PASS_ALL
                        total_correct += 1.0
                    if v != self.FAILED:
                        total_complied += 1.0
        print("\tAll Correct Rate:{0}".format(total_correct / self.n))
        print("\tAll Compile Rate:{0}".format(total_complied / self.n))        
                